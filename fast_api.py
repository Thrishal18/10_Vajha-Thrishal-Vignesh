import os
import asyncio
from typing import Dict, List
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Import Chains (adjust based on your version, using the ones that worked for you)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Optimization & Retry Logic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

# 1. Setup & Config
load_dotenv()
app = FastAPI(title="RAG Chatbot (Optimized)")

# Store chat history in memory
store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 2. Optimized Retrieval
# Using MiniLM (Fast CPU embeddings)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setup Chroma (Cloud)
vector_store = Chroma(
    collection_name="my_cloud_collection",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

# PERFORMANCE FIX: Reduce k to 2 (only get top 2 docs instead of 4)
# This significantly reduces the data sent to Gemini, speeding up response.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# 3. Rate-Limited LLM Wrapper
# This prevents the app from crashing on "Too Many Requests"
class RateLimitedGemini(ChatGoogleGenerativeAI):
    @retry(
        retry=retry_if_exception_type(ResourceExhausted), # Only retry on 429/Quota errors
        stop=stop_after_attempt(5),                       # Try 5 times
        wait=wait_exponential(multiplier=2, min=2, max=10) # Wait 2s, 4s, 8s...
    )
    async def _agenerate(self, *args, **kwargs):
        # We override the async generate method to include retries
        return await super()._agenerate(*args, **kwargs)

# Initialize Optimized LLM
llm = RateLimitedGemini(
    model="gemini-2.5-flash",  # Updated to current model
    temperature=0.3,
    convert_system_message_to_human=True,
    max_retries=5
)

# 4. Create RAG Chains

# A. History Aware Retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# B. QA Chain
qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "cannot answer and suggest escalating to a human agent. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# C. Final Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 5. Pydantic Models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class EscalateRequest(BaseModel):
    session_id: str

# 6. API Routes

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint. 
    Uses 'ainvoke' for async performance.
    """
    try:
        # PERFORMANCE FIX: Use 'ainvoke' (Async) instead of 'invoke'
        # This allows the server to handle other requests while waiting for Gemini
        response = await rag_chain_with_history.ainvoke(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}},
        )
        
        bot_answer = response["answer"]
        escalation_suggested = False
        
        # Heuristic check for escalation
        if "escalat" in bot_answer.lower() or "cannot answer" in bot_answer.lower():
            escalation_suggested = True

        return {
            "response": bot_answer,
            "escalation_suggested": escalation_suggested
        }

    except Exception as e:
        print(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/escalate")
async def escalate_endpoint(request: EscalateRequest):
    """
    Summarizes conversation for human handover.
    """
    if request.session_id not in store:
        raise HTTPException(status_code=404, detail="Session ID not found")

    history = store[request.session_id].messages
    
    if not history:
        return {"summary": "No conversation history to summarize."}

    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history])

    summary_prompt = (
        f"Please summarize the following conversation for a human support agent. "
        f"Highlight the user's main issue and why the bot failed to resolve it.\n\n"
        f"Conversation:\n{conversation_text}"
    )

    try:
        # Also using async invoke for the summary
        summary_response = await llm.ainvoke(summary_prompt)
        
        return {
            "status": "Escalated",
            "summary": summary_response.content,
            "full_history_count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
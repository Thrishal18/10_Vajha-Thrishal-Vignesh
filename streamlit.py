import streamlit as st
import requests
import uuid

# Configuration
API_URL = "http://127.0.0.1:8000"  # Ensure this matches your FastAPI port
st.set_page_config(page_title="RAG Support Agent", page_icon="🤖")

# 1. Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with the documents today?"}
    ]

# 2. Sidebar Controls
with st.sidebar:
    st.header("⚙️ Controls")
    st.write(f"**Session ID:** `{st.session_state.session_id}`")
    
    if st.button("🗑️ Reset Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [
            {"role": "assistant", "content": "Session reset. How can I help?"}
        ]
        st.rerun()

    st.divider()
    
    # Manual Escalation Button
    if st.button("🚨 Escalate to Human"):
        with st.spinner("Summarizing conversation for human agent..."):
            try:
                response = requests.post(
                    f"{API_URL}/escalate", 
                    json={"session_id": st.session_state.session_id}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success("Request Escalated Successfully!")
                    st.markdown(f"### 📝 Agent Summary:\n{data['summary']}")
                    st.info(f"Total messages analyzed: {data.get('full_history_count', 0)}")
                else:
                    st.error(f"Failed to escalate: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# 3. Chat Interface
st.title("🤖 Intelligent RAG Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt
                    }
                )
                
            if response.status_code == 200:
                data = response.json()
                bot_answer = data["response"]
                is_escalation_suggested = data.get("escalation_suggested", False)
                
                # Display bot response
                message_placeholder.markdown(bot_answer)
                st.session_state.messages.append({"role": "assistant", "content": bot_answer})
                
                # Auto-Suggestion for Escalation
                if is_escalation_suggested:
                    st.warning("⚠️ The bot seems unsure. You might want to escalate this to a human agent using the sidebar.")
            
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                message_placeholder.error(error_msg)
                
        except requests.exceptions.ConnectionError:
            message_placeholder.error("❌ Could not connect to the backend. Is FastAPI running?")
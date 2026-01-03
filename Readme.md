# 📞 AI Customer Service Agent (RAG over Tickets & Dialogues)

> **A Retrieval-Augmented Generation (RAG) assistant built to automate telecommunications support by learning from historical customer-agent interactions and support tickets.**

---

## 📖 Table of Contents
- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Datasets](#-datasets)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)

---

## 🎯 Problem Statement
Customer support centers in the telecom industry face high latency and inconsistent responses due to the sheer volume of repetitive queries (e.g., billing issues, network outages). Human agents spend excessive time searching for solutions, and scaling support teams is costly.

**The Goal:** Build a knowledge assistant that can instantly answer common queries by retrieving context from historical logs while knowing when to escalate complex issues to human agents.

## 💡 Solution Overview
This project implements a **RAG (Retrieval-Augmented Generation)** pipeline. It ingests transcripts of past agent-customer conversations and support tickets to build a vector knowledge base. When a user asks a question, the system retrieves the most relevant past solutions to generate an accurate response.

## ✨ Key Features
* **Knowledge Retrieval:** Uses Vector Search (ChromaDB/FAISS) to find semantically similar past issues.
* **Source Citations:** The `/ask` endpoint returns the generated answer along with the `Source IDs` of the documents used to ground the response.
* **Escalation Protocol:** Includes logic to detect negative sentiment or complex technical keywords, suggesting an "Escalation to Human Agent" when the AI cannot confidently answer.
* **Context-Aware:** Understands telco-specific terminology (e.g., "bandwidth," "roaming," "latency").

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB or FAISS
* **LLM Integration:** OpenAI GPT / HuggingFace Hub / Ollama
* **API Framework:** FastAPI (or Flask)

## 📊 Datasets
The model is powered by the following open-source datasets:

1.  **Primary:** [Telecom Agent–Customer Interaction Text](https://www.kaggle.com/datasets/avinashok/telecomagentcustomerinteractiontext)
    * *Used for:* Learning dialogue flows and conversational answers.
2.  **Secondary:** [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
    * *Used for:* Structured problem-solution mapping.

---


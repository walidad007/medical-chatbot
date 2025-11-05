# 🧠 MediMind – AI Medical Chatbot (RAG)

**MediMind** is an AI-powered medical assistant built with **LangChain**, **FAISS**, and **Groq LLM**.  
It answers health-related queries from curated medical documents using **Retrieval-Augmented Generation (RAG)** —  
providing helpful, context-aware responses from verified information sources.

> ⚠️ **Disclaimer:**  
> This chatbot is **not a medical professional**. It provides information for educational purposes only and must not replace medical consultation.

---

## 🌟 Project Overview

**MediMind** demonstrates how to build an **AI Medical Chatbot** capable of:
- Loading and processing PDFs as a knowledge base  
- Creating vector embeddings using Hugging Face models  
- Storing vectors in a FAISS database for efficient retrieval  
- Using **LangChain** to connect document retrieval with a **Groq LLM**  
- Providing a user-friendly chat interface built with **Streamlit**

This project serves as a prototype for healthcare-related assistants, document search tools, or domain-specific chatbots.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Programming Language** | Python 3.10+ |
| **Framework** | LangChain |
| **Vector Database** | FAISS |
| **Embeddings** | Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) |
| **LLM** | Groq (can be replaced with OpenAI or Hugging Face APIs) |
| **Frontend** | Streamlit |
| **Environment** | `.env` + `python-dotenv` |

---

## 🗂️ Repository Structure

medical-chatbot-main/
├── data/ # PDFs used as knowledge base
│ └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── medibot.py # Streamlit app (frontend + RAG pipeline)
├── create_memory_for_llm.py # Builds FAISS vector DB from PDFs
├── connect_memory_with_llm.py # Command-line RAG demo
├── requirements.txt # Project dependencies
├── pyproject.toml # Optional project setup
└── README.md # Project documentation


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/walidad007/medical-chatbot-main.git
cd medical-chatbot-main

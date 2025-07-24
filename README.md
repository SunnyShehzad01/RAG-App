# 🤖 PDF Q&A Chatbot using RAG (Gemini 2.0 + HuggingFace + FAISS)

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using:

- Google Gemini 2.0 (via `google.generativeai`)
- HuggingFace sentence-transformers for generating embeddings
- FAISS for vector storage and retrieval
- Streamlit for the interactive chatbot interface

## 📌 Features

- Upload any **PDF document**
- Automatically chunk and embed the content
- Retrieve context using FAISS vector search
- Ask natural language questions and get AI-generated answers using Google Gemini 2.0
- Minimal UI built with Streamlit

---

## 🧠 How it Works

1. **PDF Upload**: You upload a document using the Streamlit interface.
2. **Text Extraction**: Pages are read using `pypdf` and text is extracted.
3. **Chunking**: The document is split into overlapping chunks using `langchain.text_splitter`.
4. **Embedding**: Chunks are converted into embeddings using HuggingFace's `all-MiniLM-L6-v2` model.
5. **Vector DB**: Embeddings are stored in a FAISS index.
6. **Retrieval**: On user query, the top-matching chunks are retrieved.
7. **Generation**: The query and retrieved context are sent to Gemini 2.0 to generate the final answer.

---

## 🚀 Quick Start

### 🔧 Requirements

Install the dependencies using pip:

```bash
pip install -r requirements.txt
streamlit run app.py

📄 RAG-based PDF Q&A Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot that allows you to upload a PDF and ask questions about its content. It uses Hugging Face's flan-t5-base model for generation and ChromaDB for storing document embeddings.

🔧 Features

Upload any PDF

Extract and chunk text using langchain

Create vector embeddings with all-MiniLM-L6-v2

Store vectors with Chroma

Answer questions using a local flan-t5-base model

Interactive Gradio interface


📁 Project Structure
rag-pdf-chatbot/
├── app/
│   └── ui.py                  # Gradio frontend
├── backend/
│   ├── rag_pipeline.py       # Core RAG logic
│   └── llm_config.py         # LLM and embedding setup (if needed)
├── data/                     # PDF uploads
├── embeddings/               # ChromaDB vector store
├── .gitignore
├── requirements.txt
└── README.md


🤖 Tech Stack

LangChain

HuggingFace Transformers

ChromaDB


🙌 Credits

Built with ❤️ by WithZahir

Gradio

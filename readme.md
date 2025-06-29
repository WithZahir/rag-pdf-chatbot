
# 🤖 RAG-based PDF Q&A Chatbot

Ask questions about your PDF using a local language model with LangChain + Gradio.



## 🚀 Features
- 📄 Upload any PDF
- 🔍 Extracts and chunks text
- 🧠 Embeds with `MiniLM-L6-v2`
- 💾 Stores in ChromaDB
- 🗣️ Answers with `Flan-T5`
- 🖼️ Clean UI with Gradio

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-qna.git
cd rag-qna
```

### 2. Create Python Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
python app/ui.py
```
Open your browser to [http://localhost:7860](http://localhost:7860)

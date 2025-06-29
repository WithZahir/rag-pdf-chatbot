import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os

# Set directories
DATA_DIR = "data"
VECTOR_DIR = "embeddings/vector_store"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load local LLM (Flan-T5)
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# --- Define RAG logic ---
def process_pdf_and_query(pdf_file, user_question):
    # Save uploaded file
    file_path = pdf_file.name 

    # Load PDF and split
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create vector DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DIR
    )
    retriever = vectorstore.as_retriever()

    # RAG pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Ask question
    response = qa_chain.invoke(user_question)
    answer = response["result"]

    # Get source previews
    sources = "\n\n".join([doc.page_content[:300] for doc in response["source_documents"]])
    
    return answer, sources

# --- Gradio UI ---
iface = gr.Interface(
    fn=process_pdf_and_query,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Ask a question")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Source Preview")
    ],
    title="📄 RAG-based PDF Q&A Chatbot",
    description="Upload a PDF and ask questions about its content!"
)

if __name__ == "__main__":
    iface.launch()

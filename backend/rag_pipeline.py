def process_pdf_and_query(pdf_file_path, query):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from transformers import pipeline
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import RetrievalQA

    # Use the file path directly
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embedding and vector store
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever()

    # Load LLM
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Build RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain.invoke(query)
    answer = response["result"]
    sources = "\n\n".join([doc.page_content[:300] for doc in response["source_documents"]])
    
    return answer, sources

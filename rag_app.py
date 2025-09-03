import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("📄 Local RAG App with OpenAI")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create embeddings & store in Chroma
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")

    # Build retriever + QA chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Ask a question
    query = st.text_input("Ask a question about your document:")
    if query:
        answer = qa.invoke(query)
        st.write("💡 Answer:", answer)

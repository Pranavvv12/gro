import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings  # Free Hugging Face Embedding

load_dotenv()

# Load the Groq API key from .env
groq_api_key = os.getenv('GROQ_API_KEY')

# Streamlit title
st.title("Smart Chatbot with Llama 2 for Accurate, Document-Based Question Answering")

# Initialize Groq model (Llama3)
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template(""" 
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
""")

# Ensure vector embedding is initialized before accessing it
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("./data/us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Load Documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        # Use HuggingFaceEmbeddings (free model)
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Example of a free model
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embedding)  # Use FAISS for embeddings

# Input text field for user query
input_prompt = st.text_input("Enter Your Question From Documents")

# Button for embedding documents
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("FAISS Database is ready")

import time

# Process input query if available
if input_prompt:
    if "vectors" not in st.session_state:
        st.error("Please click 'Documents Embedding' first to initialize the vector database.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()

        response = retrieval_chain.invoke({'input': input_prompt})

        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # Display document similarity search results in an expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")



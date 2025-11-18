import os
import openai
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Load API keys and env vars
load_dotenv("resources/properties.env")
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
openai.base_url = "https://openrouter.ai/api/v1"

# Streamlit config
st.set_page_config(page_title="Q&A ChatBot", layout="wide")
st.title("Q&A ChatBot")

with st.sidebar:
    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"/"assistant", "content": "..."}

# If a file is uploaded
if uploaded_file:
    # Extract and chunk PDF content
    reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)

    # Generate vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Initialize LLM
    llm = ChatOpenAI(
        openai_api_key=openai.api_key,
        temperature=0,
        max_tokens=1000,
        model_name="openai/gpt-3.5-turbo",
        openai_api_base="https://openrouter.ai/api/v1"
    )
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything about the uploaded PDF...")

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Perform vector search and answer
        with st.spinner("Thinking..."):
            similar_docs = vector_store.similarity_search(user_input)
            response = qa_chain.run(input_documents=similar_docs, question=user_input)

        # Append assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

else:
    st.info("👈 Upload a PDF file from the sidebar to get started.")

import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# NEW OpenAI SDK
from openai import OpenAI

# LangChain Components
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough


# 🔹 Environment Variables
load_dotenv("resources/properties.env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Streamlit UI
st.set_page_config(page_title="Q&A ChatBot", layout="wide")
st.title("Q&A ChatBot")

with st.sidebar:
    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


# Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []


# If PDF uploaded
if uploaded_file:
    # Extract PDF text
    reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # LLM for answers
    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        openai_api_key=client.api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    # ❗ LCEL RAG PIPELINE (new chain system)
    def combine_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": combine_docs, "question": RunnablePassthrough()}
        | llm
    )

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    user_input = st.chat_input("Ask me anything about the uploaded PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            similar_docs = vector_store.similarity_search(user_input)

            # RAG Call
            
            response = rag_chain.invoke({
                "context": combine_docs(similar_docs),
                "question": user_input
            })

        st.session_state.messages.append({"role": "assistant", "content": response.content})

        with st.chat_message("assistant"):
            st.markdown(response.content)

else:
    st.info("👈 Upload a PDF file from the sidebar to get started.")

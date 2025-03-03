# MUST BE FIRST STREAMLIT COMMAND
import streamlit as st
st.set_page_config(
    page_title="WebChat Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IMPORTS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
import os
import re

# ========================
# CORE FUNCTIONS
# ========================

def get_vectorstore_from_url(url):
    """Loads website content and creates a vector store for retrieval."""
    try:
        with st.spinner("Processing website content..."):
            loader = WebBaseLoader(url)
            document = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                separators=["\n\n", "\n", "(?<=\. )", " "]
            )
            document_chunks = text_splitter.split_documents(document)

            # Debugging: Show sample document chunks
            st.write("Sample document chunks:", [chunk.page_content[:200] for chunk in document_chunks[:3]])

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            vector_store = Chroma.from_documents(
                document_chunks,
                embeddings,
                persist_directory="./chroma_db"
            )
            return vector_store
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        st.stop()

def get_response(user_input):
    """Generates and cleans a response based on user input."""
    try:
        if "vector_store" not in st.session_state:
            st.error("Vector store not initialized. Please enter a website URL.")
            return "Error: Vector store not initialized."

        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={
                "temperature": 0.4,
                "max_new_tokens": 300,
                "top_p": 0.9
            }
        )

        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(user_input)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a professional analyst. Provide a concise, paragraph-length answer (2-3 sentences) based on the main content of the website, ignoring tags, footers, or metadata.
        Context: {context}
        Question: {user_input}"""

        response = llm.generate([prompt])
        cleaned_response = clean_response(response.generations[0][0].text)
        return cleaned_response.strip()

    except Exception as e:
        return f"Error: {str(e)}"

def clean_response(text):
    """Cleans the generated response to remove irrelevant content."""
    text = re.sub(r'System:.*?Context:', '', text, flags=re.DOTALL)
    text = re.sub(r'(Tags By LangChain|Join our newsletter|Subscribe).*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

# ========================
# UI COMPONENTS
# ========================

def main_ui():
    """Handles the Streamlit UI and user interactions."""
    # (UI code remains the same as before)

# ========================
# RUN THE APP
# ========================

if __name__ == "__main__":
    main_ui()
# MUST BE FIRST STREAMLIT COMMAND
import streamlit as st
st.set_page_config(
    page_title="WebChat+",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
import os
import re

load_dotenv()

#for verifying token
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error("Please set your HUGGINGFACEHUB_API_TOKEN in the .env file")
    st.stop()



## Functions for processing website content and generating responses
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

def get_context_retriever_chain(vector_store):
    """Sets up a retriever chain to fetch relevant documents based on chat history."""
    try:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 100,
                "top_p": 0.9
            }
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Prompt to generate a specific search query
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Generate a search query to find the main explanation of the topic in the website content.")
        ])
        
        return create_history_aware_retriever(llm, retriever, prompt)
    except Exception as e:
        st.error(f"Error creating retriever chain: {str(e)}")
        st.stop()

def get_conversational_rag_chain(retriever_chain):
    """Creates a conversational RAG chain for generating responses."""
    try:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={
                "temperature": 0.4,
                "max_new_tokens": 300,
                "top_p": 0.9
            }
        )
        
        #prompt update for conciseness
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional analyst. Provide a concise, paragraph-length answer (2-3 sentences) based on the main content of the website, ignoring tags, footers, or metadata."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Context: {context}\n\nQuestion: {input}"),
        ])
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever_chain, document_chain)
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        st.stop()

def clean_response(text):
    text = re.sub(r'System:.*?Context:', '', text, flags=re.DOTALL)
    text = re.sub(r'(Tags By LangChain|Join our newsletter|Subscribe).*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def get_response(user_input):
    try:
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        
        # Debugging: Show generated search query
        search_query = retriever_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        st.write(f"Generated search query: {search_query}")
        
        # Debugging: Show retrieved documents
        retrieved_docs = retriever_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        st.write("Retrieved documents:", [doc.page_content[:200] for doc in retrieved_docs])
        
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        
        cleaned_response = clean_response(response['answer'])
        return cleaned_response.strip()
    except Exception as e:
        return f"Error: {str(e)}"



## Display the main UI
def main_ui():
    """Handles the Streamlit UI and user interactions."""
    # Header Section
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=150)
    with col2:
        st.title("WebChat+")
        st.caption("Your Professional Web Assistant by Abenezer Tesfaye")

    #sidebar configg
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:2rem;">
            <img src="https://cdn-icons-png.flaticon.com/512/1496/1496187.png" width="80" style="margin-bottom:1rem;">
            <h3 style="color:#3498db; margin:0;">Settings Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        website_url = st.text_input(
            "Website URL",
            key="website_url",
            placeholder="https://example.com"
        )
        
        st.markdown("---")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.markdown("**System Status**")
            st.markdown(f'<span style="color:#2ecc71;">●</span> Operational', unsafe_allow_html=True)
        with status_col2:
            st.markdown("**API Status**")
            if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
                st.markdown(f'<span style="color:#2ecc71;">●</span> Connected', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="color:#e74c3c;">●</span> Disconnected', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("ℹ️ **Tip:** For best results, use content-rich websites with clear text structure.")

    # Main logic
    if not website_url:
        st.info("🌐 Please enter a website URL to begin your conversation")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! I'm ready to help you analyze this website. Ask me anything about its content!")
            ]
        
        if "vector_store" not in st.session_state:
            with st.status("🧠 Processing website content...", expanded=True) as status:
                st.write("✓ Connecting to knowledge base")
                st.session_state.vector_store = get_vectorstore_from_url(website_url)
                st.write("✓ Building semantic index")
                status.update(label="Processing complete!", state="complete")

        # Chat display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                avatar = "🤖" if isinstance(message, AIMessage) else "👤"
                with st.chat_message("assistant" if isinstance(message, AIMessage) else "user", 
                                   avatar=avatar):
                    st.markdown(f"""
                    <div style="
                        padding: 1rem;
                        border-radius: 10px;
                        background: {'#e3f2fd' if isinstance(message, AIMessage) else '#f5f5f5'};
                    ">
                        {message.content}
                    </div>
                    """, unsafe_allow_html=True)

        # User input
        user_query = st.chat_input("💬 Type your message here...", key="user_input")
        if user_query:
            with st.spinner("🔍 Analyzing content..."):
                response = get_response(user_query)
            
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            st.rerun()

if __name__ == "__main__":
    main_ui()
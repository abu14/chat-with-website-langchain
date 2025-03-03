import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from apikey import openai_api_key

os.environ['OPENAI_API_KEY'] = openai_api_key

# Load data
loader = PyPDFLoader("/Users/neilmcdevitt/VSCode Projects/Cashvertising-Free-PDF-Book.pdf")
pages = loader.load_and_split()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
)

# Embeddings
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    print(page_contents_array)
    return page_contents_array

# LLM model and memory
llm = ChatOpenAI(temperature=.2, model="gpt-4-turbo-preview", max_tokens=650)
memory = ConversationBufferWindowMemory(k=5)
conversation = ConversationChain(
    llm=llm, verbose=True, memory=memory
)

# Display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Cashvertising")

# Function to format chat history for the template
def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_history += f"You: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"AI: {message.content}\n"
    return formatted_history

# Function to get response
def get_response(query, chat_history):
    formatted_chat_history = format_chat_history(chat_history)
    template = f"""
    Your specialized prompt template here...

    Chat history: {formatted_chat_history}

    User question: {query}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "chat_history": formatted_chat_history,
        "user_question": query
    })

# Conversation display
for message in st.session_state['chat_history']:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("your message")
if user_query is not None and user_query != "":
    human_message = HumanMessage(user_query)
    st.session_state['chat_history'].append(human_message)
    
    ai_response = get_response(user_query, st.session_state['chat_history'])
    ai_message = AIMessage(ai_response.content if ai_response else "I'm not sure, could you rephrase?")
    
    st.session_state['chat_history'].append(ai_message)

    with st.chat_message("AI"):
        st.markdown(ai_message.content)```
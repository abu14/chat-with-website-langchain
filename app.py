# pip install streamlit langchain-core langchain-community beautifulsoup4 

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader

def get_response(user_input):
    return "I not able to answer that"

def get_vectorscore_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

#page config
st.set_page_config(page_title="Chat with Any Website",page_icon="🧊")
st.title("Chat with Websites")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, how can I help you?"),
        ]

#side bar 
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter website URL")
    st.write("Example: https://www.google.com")

if website_url is None or website_url == "":
    st.info("Please enter a website URL, before starting the chat")
else:
    documents = get_vectorscore_from_url(website_url)
    with st.sidebar:
        st.write(documents)
    #user inputs 
    user_query = st.chat_input("Type your questions here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    #chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
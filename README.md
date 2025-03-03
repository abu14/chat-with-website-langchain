# WebChat+ - Chat with Websites from URL


![GitHub contributors](https://img.shields.io/github/contributors/abu14/chat-with-website-langchain)
![GitHub forks](https://img.shields.io/github/forks/abu14/chat-with-website-langchain?style=social)
![GitHub stars](https://img.shields.io/github/stars/abu14/chat-with-website-langchain?style=social)
![GitHub issues](https://img.shields.io/github/issues/abu14/chat-with-website-langchain)
![GitHub license](https://img.shields.io/github/license/abu14/chat-with-website-langchain)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/abenezer-tesfaye-191579214/)

WebChat+ is an AI-powered web assistant that I built that allows anyone to interact with and extract meaningful insights from website and it's content. Built using Streamlit, LangChain, and Huggingface, WebChat+ leverages state-of-the-art language model from Mistral (Mistral-7B-Instruct-v0.2) and embeddings to facilitate intelligent conversations based on retrieved website data.

## 🚀 Features

- **Website Content Processing**: Extracts and tokenizes website content for efficient retrieval.
- **Conversational AI**: Uses a retrieval-augmented generation (RAG) model to answer user queries based on website data.
- **Hugging Face Integration**: Utilizes models from Hugging Face Hub for intelligent response generation.
- **Vector Store for Retrieval**: Employs ChromaDB for semantic search and fast document retrieval.
- **Streamlit UI**: Provides a user-friendly and interactive chat experience.
- **Real-Time API Status Check**: Displays system and API connection status.
- **Customizable Settings**: Supports dynamic website input and personalized interaction.


## 🛠️ Installation & Setup
#### Prerequisites

```
Python 3.8+
pip
A Hugging Face API token 
```

#### Clone the Repository
```
git clone https://github.com/abu14/chat-with-website-langchain
cd WebChat+
```

#### Requirements
```
pip install -r requirements.txt
```

#### Set Up Environment Variables
Create a ```.env``` file in the root directory and add your Hugging Face API token:
```
HUGGINGFACEHUB_API_TOKEN=your_api_token_here
```


## Tech Stack
<p>
<img src="https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/-LangChain-1E88E5?style=flat&logo=langchain&logoColor=white">
<img src="https://img.shields.io/badge/-Hugging%20Face-FFBF00?style=flat&logo=huggingface&logoColor=white">
<img src="https://img.shields.io/badge/-ChromaDB-009688?style=flat&logo=databricks&logoColor=white">
<img src="https://img.shields.io/badge/-OpenAI-412991?style=flat&logo=openai&logoColor=white">
<img src="https://img.shields.io/badge/-HTML-E34F26?style=flat&logo=html5&logoColor=white">
<img src="https://img.shields.io/badge/-BeautifulSoup-181717?style=flat&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-Requests-0052CC?style=flat&logo=python&logoColor=white">
</p>

```
Frontend: Streamlit
Language Models: Hugging Face Hub (e.g., Mistral-7B-Instruct-v0.2)
Embeddings: Hugging Face Embeddings (e.g., sentence-transformers/all-MiniLM-L6-v2)
Vector Store: Chroma
Text Processing: LangChain, BeautifulSoup, RecursiveCharacterTextSplitter
Environment Management: dotenv
```

## 🔧 Usage

#### Run the Streamlit app
Once the application starts, follow these steps:
```
1. Enter a website URL in the sidebar.
2. Wait for content processing to complete.
3. Ask questions related to the website content.
```

## 📌 How It Works

- **Content Loading**: Extracts text from the given website URL.
- **Text Splitting**: Breaks the content into manageable chunks for better retrieval.
- **Embedding Generation**: Converts text into vector embeddings using Hugging Face models.
- **Vector Storage & Retrieval**: Stores the embeddings in ChromaDB for quick semantic search.
- **Conversational RAG Model**: Uses LangChain to retrieve relevant content and generate responses.
- **Response Processing**: Filters and cleans responses before displaying them in chat

<p align="center">
  <img src="assets/HTML-rag-diagram1.PNG" alt="Project Workflow" width="900" height="400">
</p>

## 📷 Screenshot
<p align="center">
  <img src="assets/webchat_project_v1.3.PNG" alt="Project Workflow" width="900" height="500">
</p>


## 📞 Contact

Developed by Abenezer Tesfaye

- GitHub: [abu14](https://github.com/abu14/chat-with-website-langchain)
- Email: tesfayeabenezer64@gmail.com





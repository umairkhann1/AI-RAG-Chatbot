import os
from dotenv import load_dotenv
import streamlit as st # type: ignore[import]

from langchain_openai import ChatOpenAI # type: ignore[import]
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

# Load env
load_dotenv()

# Page title
st.title("🤖 AI Document Chatbot")

# API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Load FAISS model
if not os.path.exists("models/index.faiss"):
    st.error("❌ FAISS model not found. Run with_faiss.py first.")
    st.stop()

faiss_index = FAISS.load_local(
    "models",
    embeddings,
    allow_dangerous_deserialization=True
)



# Chat model
chat = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    temperature=0
)

# Ensure API key is set before creating the chat client
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set. Please add GROQ_API_KEY to your .env or environment.")
    st.stop()

# Input box
question = st.text_input("Ask a question:")

# When user asks question
if question:

    docs = faiss_index.similarity_search(
        query=question,
        k=2
    )

    main_content = question + "\n\n"

    for doc in docs:
        main_content += doc.page_content + "\n\n"

    messages = [
        SystemMessage(
            content="""
You are an AI assistant.
Answer ONLY from provided documents.
If answer not found say:
'Hmm, I am not sure.'
"""
        ),
        HumanMessage(content=main_content)
    ]

    ai_response = chat.invoke(messages).content

    st.write("### Answer:")
    st.write(ai_response)
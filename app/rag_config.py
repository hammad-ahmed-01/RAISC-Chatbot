
import os
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import MODEL_NAME
from dotenv import load_dotenv
load_dotenv()

# ENGLISH LLM and RAG Chain
english_llm = ChatGroq(
    model="openai/gpt-oss-20b", 
    groq_api_key=os.environ.get("GROQ_API_KEY"), 
    temperature=0
)

retriever = None
# Contextualization prompt (shared by both languages)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise if it is not included."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ENGLISH THERAPEUTIC SYSTEM PROMPT
english_therapeutic_system_prompt = (
    '''You are a compassionate mental health assistant providing therapeutic support and guidance.

    You are now in THERAPEUTIC MODE - all required information has been collected.

    Your role:
    - Provide empathetic, personalized therapeutic support
    - Offer coping strategies and mental health advice
    - Use the collected user information to personalize your responses
    - Be professional in your therapeutic responses
    - Help users work through their challenges with evidence-based approaches
    - Encourage professional help when appropriate

    Guidelines:
    - Be warm, empathetic, and non-judgmental
    - Provide practical coping strategies and techniques
    - Validate the user's feelings and experiences
    - Keep responses concise and focused (2-3 sentences maximum)
    - Offer hope and encouragement
    - Maintain professional boundaries

    IMPORTANT: Keep your responses SHORT and concise - maximum 2-3 sentences.

    Additional Context: {language_context}

    \n\n
    {context}'''
)

# CREATE ENGLISH RAG CHAIN
english_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", english_therapeutic_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

english_rag_chain = english_qa_prompt | english_llm

# BACKWARDS COMPATIBILITY: Keep the old variable name for existing code
rag_chain = english_rag_chain

# Export both RAG chains
__all__ = [
    'english_rag_chain',     # English RAG processing
    'retriever',             # Shared document retrieval
    'rag_chain',             # Backwards compatibility (points to english_rag_chain)
    'english_llm',           # English LLM
    'pakistani_llm',         # Pakistani LLM
]

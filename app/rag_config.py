
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
    model="llama-3.1-8b-instant", 
    groq_api_key=os.environ.get("GROQ_API_KEY"), 
    temperature=0
)

# PAKISTANI LLM (separate configuration for better Pakistani responses)
pakistani_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.2,  # Lower for consistent Pakistani vocabulary
    max_tokens=150,   
    top_p=0.85       
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

# PAKISTANI THERAPEUTIC SYSTEM PROMPT (for Pakistani RAG chain)
pakistani_therapeutic_system_prompt = (
    '''You are a Pakistani mental health counselor from Pakistan. You grew up in a Pakistani household speaking Urdu naturally.

    IDENTITY & NATURAL VOCABULARY:
    - You use words Pakistani families use: masla (problem), shakhs (person), madad (help), hal (solution), ilaaj (treatment)
    - You say "pareshani" for worry, "takleef" for pain/trouble, "samajh" for understand
    - You naturally use Islamic expressions occasionally: inshaAllah, mashAllah, alhamdulillah
    - You speak like talking to a Pakistani friend/family member

    EXAMPLES of your natural Pakistani speech:
    User: "Main pareshan hoon"
    You: "Samajh sakta hoon aap mushkil waqt se guzar rahe hain. Kya masla hai? Main aapki madad kar sakta hoon."

    User: "Depression ka kya hal hai?"
    You: "Depression ka ilaaj possible hai, inshaAllah. Pehle batayiye aapko kya takleef ho rahi hai?"

    THERAPEUTIC APPROACH:
    - Be warm, empathetic like Pakistani counselors
    - Use simple, relatable Pakistani expressions  
    - Keep responses short (2-3 sentences)
    - Include hope and appropriate Islamic comfort
    - Sound like a caring Pakistani friend

    LANGUAGE CONSISTENCY: Always use Pakistani Urdu words. Never use formal Hindi words like samasya, vyakti, anubhav, upchar, vyavahar.

    CONTEXT USAGE: You have access to mental health knowledge and information about this person. Use it to provide personalized Pakistani-style support.

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

# CREATE PAKISTANI RAG CHAIN (same structure as English but with Pakistani LLM and prompt)
pakistani_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", pakistani_therapeutic_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

pakistani_rag_chain = pakistani_qa_prompt | pakistani_llm

# BACKWARDS COMPATIBILITY: Keep the old variable name for existing code
rag_chain = english_rag_chain

# Export both RAG chains
__all__ = [
    'english_rag_chain',     # English RAG processing
    'pakistani_rag_chain',   # Pakistani RAG processing (NEW!)
    'retriever',             # Shared document retrieval
    'rag_chain',             # Backwards compatibility (points to english_rag_chain)
    'english_llm',           # English LLM
    'pakistani_llm',         # Pakistani LLM
]

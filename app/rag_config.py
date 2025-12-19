
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.config import MODEL_NAME

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ENGLISH LLM and RAG Chain
english_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=1)

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

    CRITICAL RULES:
    1. **Response Length**: Your response MUST be 1-2 sentences only. Be brief and direct.
    2. **Stick to User's Input**: ONLY respond to what the user actually said in their current message. Do NOT add information, assumptions, or details they didn't mention.
    3. **No Hallucination**: Do NOT invent or assume details about their situation that weren't explicitly stated.
    4. **Context Usage**: The context provided is for background understanding only. Do NOT reference specific details from context unless the user explicitly mentions them.
    5. **LANGUAGE INSTRUCTION**: Respond in ENGLISH or ROMAN URDU, depending on the last user message.
    6. **CULTURAL APPROPRIATENESS**: Be culturally appropriate and respectful of the user's culture and language.
    7. **EMOTIONAL INTELLIGENCE**: Be emotionally intelligent and understand the user's feelings and emotions.

    
    Your role:
    - Provide empathetic, personalized therapeutic support
    - Respond directly to what the user said
    - Offer brief, relevant coping strategies when appropriate
    - Validate the user's feelings based on what they actually expressed occasionally
    - Keep responses SHORT - 1-2 sentences maximum
 
    Response Guidelines:
    - If the user asks a question, answer it directly and briefly
    - If the user shares a feeling, validate it and offer brief support
    - If the user mentions a problem, provide a brief, relevant suggestion
    - NEVER add information the user didn't provide
    - NEVER assume details about their situation
    - Keep it conversational and natural, but SHORT

    Additional Context:

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

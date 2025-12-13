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
current_dir = os.path.dirname(os.path.abspath(__file__))

PERSISTENT_DIRECTORY = os.path.join(current_dir, "../db/chroma_db_with_metadata")
# Initialize Embeddings and VectorStore
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME,   encode_kwargs={'normalize_embeddings': True} )
db = Chroma(persist_directory=PERSISTENT_DIRECTORY, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# ENGLISH LLM and RAG Chain
english_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.environ.get("GROQ_API_KEY"), 
    temperature=0
)

# PAKISTANI LLM (separate configuration for better Pakistani responses)
pakistani_llm = ChatGroq(
    model="openai/gpt-oss-20b",
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

    CRITICAL RULES:
    1. **Response Length**: Your response MUST be 1-2 sentences only. Be brief and direct.
    2. **Stick to User's Input**: ONLY respond to what the user actually said in their current message. Do NOT add information, assumptions, or details they didn't mention.
    3. **No Hallucination**: Do NOT invent or assume details about their situation that weren't explicitly stated.
    4. **Context Usage**: The context provided is for background understanding only. Do NOT reference specific details from context unless the user explicitly mentions them.

    Your role:
    - Provide empathetic, personalized therapeutic support
    - Respond directly to what the user said
    - Offer brief, relevant coping strategies when appropriate
    - Validate the user's feelings based on what they actually expressed
    - Keep responses SHORT - 1-2 sentences maximum

    Response Guidelines:
    - If the user asks a question, answer it directly and briefly
    - If the user shares a feeling, validate it and offer brief support
    - If the user mentions a problem, provide a brief, relevant suggestion
    - NEVER add information the user didn't provide
    - NEVER assume details about their situation
    - Keep it conversational and natural, but SHORT

    Additional Context: {language_context}

    \n\n
    {context}'''
)


# PAKISTANI ROMAN URDU THERAPEUTIC SYSTEM PROMPT
pakistani_therapeutic_system_prompt = (
    '''You are a compassionate mental health assistant providing therapeutic support and guidance.

    **LANGUAGE INSTRUCTION**: You MUST respond in Pakistani Roman Urdu (Urdu written in Roman/Latin script).
    Use authentic Pakistani vocabulary, expressions, and cultural context.
    Do NOT use Hindi words - use only Pakistani Urdu vocabulary.

    You are now in THERAPEUTIC MODE - all required information has been collected.

    CRITICAL RULES:
    1. **Response Length**: Your response MUST be 1-2 sentences only. Be brief and direct.
    2. **Stick to User's Input**: ONLY respond to what the user actually said in their current message. Do NOT add information, assumptions, or details they didn't mention.
    3. **No Hallucination**: Do NOT invent or assume details about their situation that weren't explicitly stated.
    4. **Context Usage**: The context provided is for background understanding only. Do NOT reference specific details from context unless the user explicitly mentions them.
    5. **Language**: Use natural, conversational Pakistani Roman Urdu. Examples:
       - Use "aap" not "tum" for respect
       - Use "main" or "mein" for "I"
       - Use "kya" for "what", "kyun" for "why", "kaise" for "how"
       - Use "theek hai", "sahi hai" for "okay/alright"
       - Use "pareshani" for "problem", "takleef" for "difficulty"
       - Use Islamic greetings naturally: "salam", "Allah ka shukr", "InshaAllah", "MashAllah"
       - Use "mehsoos" for "feel", "samajh" for "understand"
       - Use "madad" for "help", "mashwara" for "advice"

    Your role:
    - Provide empathetic, personalized therapeutic support in Pakistani Roman Urdu
    - Respond directly to what the user said
    - Offer brief, relevant coping strategies when appropriate
    - Validate the user's feelings based on what they actually expressed
    - Keep responses SHORT - 1-2 sentences maximum
    - Be culturally sensitive to Pakistani norms and values

    Response Guidelines:
    - If the user asks a question, answer it directly and briefly in Roman Urdu
    - If the user shares a feeling, validate it and offer brief support in Roman Urdu
    - If the user mentions a problem, provide a brief, relevant suggestion in Roman Urdu
    - NEVER add information the user didn't provide
    - NEVER assume details about their situation
    - Keep it conversational and natural, but SHORT
    - Use appropriate Islamic expressions where culturally relevant

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

# CREATE PAKISTANI ROMAN URDU RAG CHAIN
pakistani_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", pakistani_therapeutic_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Create history-aware retriever
english_history_aware_retriever = create_history_aware_retriever(
    english_llm,
    retriever,
    contextualize_q_prompt
)
english_question_answer_chain = create_stuff_documents_chain(
    english_llm,
    english_qa_prompt
)
english_rag_chain = create_retrieval_chain(
    english_history_aware_retriever,
    english_question_answer_chain
)

pakistani_history_aware_retriever = create_history_aware_retriever(
    pakistani_llm,
    retriever,
    contextualize_q_prompt
)
pakistani_question_answer_chain = create_stuff_documents_chain(
    pakistani_llm,
    pakistani_qa_prompt
)

pakistani_rag_chain = create_retrieval_chain(
    pakistani_history_aware_retriever,
    pakistani_question_answer_chain
)

# BACKWARDS COMPATIBILITY: Keep the old variable name for existing code
rag_chain = english_rag_chain

# Export both RAG chains
__all__ = [
    'english_rag_chain',     # English RAG processing
    'pakistani_rag_chain',   # Pakistani Roman Urdu RAG processing
    'retriever',             # Shared document retrieval
    'rag_chain',             # Backwards compatibility (points to english_rag_chain)
    'english_llm',           # English LLM
    'pakistani_llm',         # Pakistani LLM
]
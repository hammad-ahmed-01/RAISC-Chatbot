import os
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import MODEL_NAME, PERSISTENT_DIRECTORY
from dotenv import load_dotenv
load_dotenv()

# Initialize Embeddings and VectorStore
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(persist_directory=PERSISTENT_DIRECTORY, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Initialize the Language Model
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.environ.get("GROQ_API_KEY"), temperature=0)

# Define Prompts for contextualization (same for both phases)
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

# SIMPLIFIED THERAPEUTIC SYSTEM PROMPT (only for therapeutic mode)
therapeutic_system_prompt = (
    '''You are a compassionate mental health assistant providing therapeutic support and guidance.

    You are now in THERAPEUTIC MODE - all required information has been collected.

    Your role:
    - Provide empathetic, personalized therapeutic support
    - Offer coping strategies and mental health advice
    - Use the collected user information to personalize your responses
    - Be thorough, caring, and professional in your therapeutic responses
    - Help users work through their challenges with evidence-based approaches
    - Encourage professional help when appropriate

    Guidelines:
    - Be warm, empathetic, and non-judgmental
    - Provide practical coping strategies and techniques
    - Validate the user's feelings and experiences
    - Use active listening techniques
    - Offer hope and encouragement
    - Maintain professional boundaries

    \n\n
    {context}'''
)

# Update prompt templates to use the simplified therapeutic prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", therapeutic_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create RAG chain for therapeutic responses only
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
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

# STRICT UNIFIED SYSTEM PROMPT
unified_system_prompt = (
    '''You are a mental health assistant with STRICT operational modes.

INFORMATION GATHERING MODE (when information is missing):
- You MUST ask for the missing information first
- DO NOT provide therapeutic advice or extensive support yet
- Keep responses to 2-3 sentences maximum
- Stay focused ONLY on getting the specific information needed
- Be warm but direct in asking for information
- NEVER mix information gathering with therapy

THERAPY MODE (when all information is collected):
- Provide full therapeutic support and guidance
- Use collected information to personalize responses
- Offer coping strategies, emotional support, and mental health advice
- Be thorough and empathetic in your therapeutic responses

STRICT RULES:
- You can ONLY be in ONE mode at a time
- NEVER provide therapy while information is missing
- NEVER skip asking for required information
- Follow the context instructions exactly
- Keep information gathering responses brief and focused

\n\n
{context}'''
)

# Update prompt templates to use the unified prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", unified_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create unified RAG chain
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


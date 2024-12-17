import os
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import MODEL_NAME, GROQ_API_KEY, PERSISTENT_DIRECTORY

# Initialize Embeddings and VectorStore
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(persist_directory=PERSISTENT_DIRECTORY, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Initialize the Language Model
llm = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)

# Define Prompts
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

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise. "
    "At the end, add a line asking if the user needs more information "
    "or would like to continue the conversation."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create History-Aware Retriever and RAG Chain
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

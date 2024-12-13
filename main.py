# This code consists of 2 parts, 1 part is using the already made vector store(our knowlegde base, AKA backbone of RAG)
# this part will answer user questions depending on the knowlegde in the vector store
# 2nd part is the user-ai interaction which will be stored in chromadb for vectorstore
# and fetched when the user comes back for context

# from dotenv import load_dotenv
import os
from google.cloud import firestore
from langchain_groq import ChatGroq
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings import HuggingFaceEmbeddings

# Define the persistent directory( to be used by the RAG)
current_dir = os.path.dirname(os.path.abspath(__file__))
# "db" is folder "chroma_db_with_metadata" is file inside the folder
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Setup Firebase Firestore
PROJECT_ID = "raisc-20b3d"
SESSION_ID = "user_4session"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_history"

# Define the embedding model
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# Initializing the LLM with chatgroq
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    #is ko hide karnay wala scene krdin :)
    groq_api_key="REMOVED_GROQ_API_KEY",
    temperature=0
)

# Contextualizing question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise if it is not included."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Creating a history aware retriever here which
# Combines the LLM and retriever(RAG) with a prompt to reframe user questions in the context of prior chat history.
# This ensures the retriever fetches contextually relevant documents even when the user's query is ambiguous or incomplete.
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
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

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#A chain to combine documents for question answering
# This chain uses the LLM and the question-answering prompt to synthesize 
# information from the retrieved documents and generate concise answers.
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#Create a RAG chai
# Combines the history-aware retriever and the question-answering chain.
# This allows the system to retrieve relevant context from the vector store
# and generate answers that are grounded in the retrieved information.
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

def chat_with_rag():
    print("Start chatting with the AI. Type 'exit' to quit.")

    while True:
        human_input = input("User: ")
        if human_input.lower() == "exit":
            break

        # Add user message to Firestore chat history
        chat_history.add_user_message(human_input)

        # Process the query through the RAG chain
        result = rag_chain.invoke({
            "input": human_input, 
            "chat_history": chat_history.messages
        })

        ai_response = result['answer']

        # Add AI response to Firestore chat history
        chat_history.add_ai_message(ai_response)

        print(f"AI: {ai_response}")

if __name__ == "__main__":
    chat_with_rag() 
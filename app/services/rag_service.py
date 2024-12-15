from app.rag_config import rag_chain

def process_user_message(user_message: str, chat_history: list) -> str:
    """
    Process a user's message through the RAG model and return the response.
    """
    result = rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history
    })

    return result["answer"]

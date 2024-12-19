from app.rag_config import rag_chain

def process_user_message(user_message: str, chat_history: list, user_data: dict = None) -> str:
    """
    Process a user's message through the RAG model and return the response.
    """
    user_context = (
        f"User information: {user_data}. "
        "This includes their mental state and history. "
        "Personalize your responses using this data."
    ) if user_data else ""

    result = rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": user_context,
    })

    return result["answer"]

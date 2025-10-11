from app.utils import get_firestore_client
from app.config import COLLECTION_NAME
from dotenv import load_dotenv
load_dotenv()
client = get_firestore_client()

def get_chat_history(session_key: str):
    """
    Retrieve the chat history for a session from Firestore.
    If the session does not exist, return an empty list.
    """
    chat_history_ref = client.collection(COLLECTION_NAME).document(session_key)
    chat_history_doc = chat_history_ref.get()

    if chat_history_doc.exists:
        return chat_history_doc.to_dict().get("messages", [])
    
    # Return an empty list if no history exists
    return []


def save_chat_history(session_key: str, chat_history: list):
    """
    Save the chat history for a session to Firestore.
    """
    chat_history_ref = client.collection(COLLECTION_NAME).document(session_key)
    chat_history_ref.set({"messages": chat_history})
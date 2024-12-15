from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.rag_service import process_user_message

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Handle real-time chat interactions with the user.
    """
    session_key = request.session_key
    user_message = request.message

    # Retrieve chat history (empty if session does not exist)
    chat_history = get_chat_history(session_key)

    # Add user's message to chat history
    chat_history.append({"role": "user", "content": user_message})

    # Generate AI response
    ai_response = process_user_message(user_message, chat_history)

    # Add AI response to chat history and save
    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history(session_key, chat_history)

    return {"response": ai_response}


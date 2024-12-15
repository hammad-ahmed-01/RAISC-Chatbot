from fastapi import APIRouter, HTTPException
from app.services.firestore_service import get_chat_history

router = APIRouter()

@router.get("/history/{session_key}")
async def get_chat_history_endpoint(session_key: str):
    """
    Retrieve chat history for a given session.
    """
    chat_history = get_chat_history(session_key)

    if not chat_history:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_key": session_key, "chat_history": chat_history}

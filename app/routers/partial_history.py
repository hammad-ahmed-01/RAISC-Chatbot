from fastapi import APIRouter, HTTPException
from app.services.firestore_service import get_chat_history

router = APIRouter()

@router.get("/history/{session_key}/{start_idx}/{end_idx}")
async def get_partial_chat_history(session_key: str, start_idx: int, end_idx: int):
    """
    Retrieve a partial chat history between start_idx and end_idx.
    """
    chat_history = get_chat_history(session_key)

    if not chat_history:
        raise HTTPException(status_code=404, detail="Session not found")

    if start_idx < 0 or end_idx >= len(chat_history) or start_idx > end_idx:
        raise HTTPException(status_code=400, detail="Invalid start or end index")

    sliced_history = chat_history[start_idx:end_idx + 1]  # +1 to include end_idx
    return {"session_key": session_key, "chat_history": sliced_history}

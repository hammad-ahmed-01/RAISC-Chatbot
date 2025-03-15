from fastapi import APIRouter
from app.models import ChatRequest
from app.services.chat_service import process_chat

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    response = await process_chat(request.session_key, request.message)
    return response
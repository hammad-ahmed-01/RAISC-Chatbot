from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_key: str
    message: str
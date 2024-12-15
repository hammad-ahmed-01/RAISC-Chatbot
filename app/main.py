from fastapi import FastAPI
from app.routers import chat, history

app = FastAPI()

# Include routers
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(history.router, prefix="/api", tags=["History"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}

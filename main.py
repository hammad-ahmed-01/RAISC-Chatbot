from fastapi import FastAPI
from app.routers import chat, history, partial_history
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
app = FastAPI()

# --- CORS (dev-friendly; tighten for prod) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://localhost:3000",# Vite (if you use it)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(history.router, prefix="/api", tags=["History"])
app.include_router(partial_history.router, prefix="/api", tags=["Partial History"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}



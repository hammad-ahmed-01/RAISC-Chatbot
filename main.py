# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # (keep if you plan to serve static assets)
from app.routers import chat, history, partial_history

app = FastAPI(title="RAISC Chatbot API")

# --- CORS (env-driven; safe defaults) ---
# Provide a comma-separated list in ALLOWED_ORIGINS env var.
# Example:
# ALLOWED_ORIGINS="https://your-frontend.com,https://web-production-deb22.up.railway.app,https://localhost:3000,http://localhost:3000"
_default_origins = [
    "https://web-production-deb22.up.railway.app",
    "https://raisc.org",
    "https://stage.raisc.org/",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5173",
    "http://localhost:3000",
    "https://localhost:3000",
]
_raw = os.getenv("ALLOWED_ORIGINS", ",".join(_default_origins))
ALLOWED_ORIGINS = sorted({o.strip().rstrip("/") for o in _raw.split(",") if o.strip()})

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(history.router, prefix="/api", tags=["History"])
app.include_router(partial_history.router, prefix="/api", tags=["Partial History"])

# Optional: mount static files if you have a ./static directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Chatbot API",
        "cors_allowed_origins": ALLOWED_ORIGINS,  # helpful for quick debugging
    }

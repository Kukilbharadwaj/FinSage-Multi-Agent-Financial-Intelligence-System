# main.py
# FastAPI application entry point for FinSage AI.

import os
from dotenv import load_dotenv

# Load .env FIRST before any other imports that might need it
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from db.database import init_db

# Create FastAPI app
app = FastAPI(
    title="FinSage AI",
    version="0.1.0",
    description="Multi-agent Indian financial assistant powered by Groq + LangGraph",
)

# CORS middleware — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.on_event("startup")
def startup():
    """Initialize database on application startup."""
    init_db()
    print("[OK] FinSage AI backend started")
    print("[API] API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

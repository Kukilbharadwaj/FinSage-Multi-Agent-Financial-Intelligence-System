# main.py
# FastAPI application entry point for FinSage AI.
# Serves the HTML/JS frontend and the REST API.
#
# This is the ONLY server you need to run:
#     python main.py
#
# MCP tools are reached over FastMCP's in-memory transport (mcp_bridge.py),
# so there is no separate MCP process, port, or startup race to manage.

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env FIRST before any other imports that might need it
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from db.database import init_db
from mcp_bridge import shutdown_mcp_runtime, startup_mcp_runtime
from observability import flush as flush_telemetry
from observability import init_langfuse

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "frontend" / "static"


def _warm_caches() -> None:
    """
    Open the RAG connections at startup rather than on the first user query.

    Retrieval needs an HTTPS session to the Hugging Face embedding API and a
    connection to Pinecone. Establishing both here means the first person to
    ask a question does not pay the handshake cost on top of their answer.
    """
    try:
        from rag.knowledge_base import warmup

        warmup()
    except Exception as exc:
        print(f"[WARN] RAG warmup skipped: {str(exc)[:120]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop application-level resources."""
    init_db()
    startup_mcp_runtime()

    # Keep startup output pure ASCII — the Windows console defaults to cp1252
    # and raises UnicodeEncodeError on characters like arrows or emoji.
    if init_langfuse():
        print(f"[OK] Langfuse telemetry active -> {os.environ.get('LANGFUSE_HOST')}")
    else:
        print("[--] Langfuse telemetry disabled")

    _warm_caches()

    print("[OK] FinSage AI backend started")
    print("[UI] Open http://localhost:8000 in your browser")
    print("[API] API docs: http://localhost:8000/docs")

    yield

    shutdown_mcp_runtime()
    flush_telemetry()


# Create FastAPI app
app = FastAPI(
    title="FinSage AI",
    version="0.4.0",
    description="Multi-agent Indian financial assistant powered by Groq + LangGraph",
    lifespan=lifespan,
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

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main UI page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    # Render (and most PaaS hosts) assign the port at runtime and expect the
    # process to bind exactly that one — a hardcoded 8000 fails their port scan
    # and the deploy is marked unhealthy. Locally PORT is unset and this stays
    # 8000, so nothing changes for development.
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

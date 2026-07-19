# app.py
# Single-entry launcher for FinSage AI.
#
# There is nothing to orchestrate any more. MCP tools are served over
# FastMCP's in-memory transport inside the backend process, so this file
# now just starts the one FastAPI server that hosts the API and the UI.
#
# Previously this booted an MCP SSE server in a background thread, polled
# an HTTP endpoint waiting for it to come up, and only then started
# uvicorn — a startup race that could time out and take the whole app
# down with it. All of that is gone.
#
# Equivalent to running:  python main.py

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("PORT") or os.getenv("BACKEND_PORT", "8000"))

    if not os.getenv("GROQ_API_KEY", "").strip():
        print("[WARN] GROQ_API_KEY is not set - the app will start, but answers will fail.")

    print(f"[OK] Starting FinSage on http://{host}:{port}")
    print(f"[UI] Open http://localhost:{port} in your browser")

    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()

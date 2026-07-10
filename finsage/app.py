import os
import multiprocessing
import threading
import time
import requests
import uvicorn

from mcp_server import mcp


# Hugging Face Spaces runs a single entry file (app.py).
# Start the local MCP SSE server in a background daemon thread,
# then launch the Gradio app in the main thread.
_MCP_STARTED = threading.Event()
_MCP_PROCESS: multiprocessing.Process | None = None
_BACKEND_STARTED = threading.Event()


def _run_mcp_server() -> None:
    mcp.settings.host = os.getenv("MCP_HOST", "0.0.0.0")
    mcp.settings.port = int(os.getenv("MCP_PORT", "7862"))

    transport = os.getenv("MCP_TRANSPORT", "sse").strip().lower()
    if transport == "http":
        mcp.run(transport="streamable-http")
        return

    mcp.run(transport="sse")


def _run_backend_server() -> None:
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = int(os.getenv("BACKEND_PORT", "8000"))
    # Use import string so startup events in main.py run normally.
    config = uvicorn.Config("main:app", host=backend_host, port=backend_port, reload=False, log_level="info")
    server = uvicorn.Server(config)
    server.run()


def _start_mcp_background() -> None:
    global _MCP_PROCESS

    if _MCP_STARTED.is_set():
        return

    run_mode = os.getenv("MCP_RUN_MODE", "thread").strip().lower()

    if run_mode == "process":
        # Prefer fork on Linux (HF Spaces). Windows spawn can recursively import app.py,
        # so process mode is automatically downgraded there.
        if os.name == "nt":
            threading.Thread(target=_run_mcp_server, daemon=True, name="mcp-sse-server").start()
        else:
            ctx = multiprocessing.get_context("fork")
            _MCP_PROCESS = ctx.Process(target=_run_mcp_server, daemon=True, name="mcp-server")
            _MCP_PROCESS.start()
    else:
        threading.Thread(target=_run_mcp_server, daemon=True, name="mcp-sse-server").start()

    _MCP_STARTED.set()


def _start_backend_background() -> None:
    if _BACKEND_STARTED.is_set():
        return

    threading.Thread(target=_run_backend_server, daemon=True, name="finsage-backend").start()
    _BACKEND_STARTED.set()


def wait_for_mcp(url: str, timeout: int = 15) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=2, stream=True)
            if response.status_code < 500:
                response.close()
                return
            response.close()
        except requests.RequestException:
            pass
        time.sleep(0.5)

    raise RuntimeError(f"MCP failed to start at {url} within {timeout}s")


def wait_for_backend(url: str, timeout: int = 25) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code < 500:
                response.close()
                return
            response.close()
        except requests.RequestException:
            pass
        time.sleep(0.5)

    raise RuntimeError(f"Backend failed to start at {url} within {timeout}s")


def build_app():
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = os.getenv("BACKEND_PORT", "8000")
    mcp_port = os.getenv("MCP_PORT", "7862")
    transport = os.getenv("MCP_TRANSPORT", "sse").strip().lower()

    default_url = f"http://127.0.0.1:{mcp_port}/mcp" if transport == "http" else f"http://127.0.0.1:{mcp_port}/sse"
    os.environ.setdefault("MCP_SERVER_URL", default_url)

    if transport == "http":
        # Current mcp_client.py uses SSE client transport. Keep this explicit so failures are clear.
        raise RuntimeError(
            "MCP_TRANSPORT=http is not supported by current mcp_client.py (SSE-only). "
            "Use MCP_TRANSPORT=sse or update mcp_client.py for Streamable HTTP client support."
        )

    # Ensure frontend talks to local FastAPI backend started below.
    os.environ.setdefault("FINSAGE_API_URL", f"http://{backend_host}:{backend_port}")

    if not os.getenv("GROQ_API_KEY", "").strip():
        print(
            "[WARN] GROQ_API_KEY is not set. App can start, but AI responses will fail until you add the secret "
            "in Hugging Face Space Settings > Variables and secrets."
        )

    _start_mcp_background()
    startup_timeout = int(os.getenv("MCP_STARTUP_TIMEOUT", "15"))
    wait_for_mcp(os.environ["MCP_SERVER_URL"], timeout=startup_timeout)

    _start_backend_background()
    backend_health_url = f"http://{backend_host}:{backend_port}/api/health"
    backend_timeout = int(os.getenv("BACKEND_STARTUP_TIMEOUT", "25"))
    wait_for_backend(backend_health_url, timeout=backend_timeout)

    # Import after env/setup so mcp_client picks the correct MCP_SERVER_URL.
    from frontend.app import create_ui, CUSTOM_CSS

    return create_ui(), CUSTOM_CSS


# Exported for Spaces runtime discovery.
demo, custom_css = build_app()



demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
        css=custom_css,
    )

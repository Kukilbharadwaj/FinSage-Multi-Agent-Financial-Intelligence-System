import os
import multiprocessing
import threading
import time
import requests
import uvicorn

from mcp_server import mcp
from main import app as fastapi_app


# Hugging Face Spaces runs a single entry file (app.py).
# Start MCP + FastAPI in background, then load Streamlit UI in this file.
_MCP_STARTED = threading.Event()
_API_STARTED = threading.Event()
_MCP_PROCESS: multiprocessing.Process | None = None


def _is_service_up(url: str, timeout: int = 2) -> bool:
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        ok = response.status_code < 500
        response.close()
        return ok
    except requests.RequestException:
        return False


def _run_mcp_server() -> None:
    mcp.settings.host = os.getenv("MCP_HOST", "0.0.0.0")
    mcp.settings.port = int(os.getenv("MCP_PORT", "8001"))

    transport = os.getenv("MCP_TRANSPORT", "sse").strip().lower()
    if transport == "http":
        mcp.run(transport="streamable-http")
        return

    mcp.run(transport="sse")


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


def _run_api_server() -> None:
    api_host = os.getenv("API_HOST", "127.0.0.1")
    api_port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(
        fastapi_app,
        host=api_host,
        port=api_port,
        reload=False,
        log_level="info",
    )


def _start_api_background() -> None:
    if _API_STARTED.is_set():
        return
    threading.Thread(target=_run_api_server, daemon=True, name="finsage-fastapi").start()
    _API_STARTED.set()


def wait_for_api(url: str, timeout: int = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)

    raise RuntimeError(f"API failed to start at {url} within {timeout}s")


def build_app():
    mcp_port = os.getenv("MCP_PORT", "8001")
    transport = os.getenv("MCP_TRANSPORT", "sse").strip().lower()

    default_url = f"http://127.0.0.1:{mcp_port}/mcp" if transport == "http" else f"http://127.0.0.1:{mcp_port}/sse"
    os.environ.setdefault("MCP_SERVER_URL", default_url)

    if transport == "http":
        # Current mcp_client.py uses SSE client transport. Keep this explicit so failures are clear.
        raise RuntimeError(
            "MCP_TRANSPORT=http is not supported by current mcp_client.py (SSE-only). "
            "Use MCP_TRANSPORT=sse or update mcp_client.py for Streamable HTTP client support."
        )

    startup_timeout = int(os.getenv("MCP_STARTUP_TIMEOUT", "15"))
    if not _is_service_up(os.environ["MCP_SERVER_URL"]):
        _start_mcp_background()
    wait_for_mcp(os.environ["MCP_SERVER_URL"], timeout=startup_timeout)

    # Start FastAPI backend (main.py) after MCP is ready.
    api_port = int(os.getenv("API_PORT", "8000"))
    api_url = f"http://127.0.0.1:{api_port}"
    os.environ.setdefault("FINSAGE_API_URL", api_url)
    if not _is_service_up(f"{api_url}/api/health"):
        _start_api_background()
    wait_for_api(f"{api_url}/api/health", timeout=int(os.getenv("API_STARTUP_TIMEOUT", "20")))

    # Import Streamlit UI only after backend is ready.
    import frontend.app  # noqa: F401

    return True


# Streamlit executes this file directly; build services then render UI.
build_app()

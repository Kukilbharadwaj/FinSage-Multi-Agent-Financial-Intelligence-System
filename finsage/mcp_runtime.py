# mcp_runtime.py
# Backend-managed MCP client runtime used by main.py.

import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

from config.settings import settings

_exit_stack: Optional[AsyncExitStack] = None
_session: Optional[ClientSession] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_tools = []


async def startup_mcp_runtime() -> None:
    """Initialize a long-lived MCP client session for backend use."""
    global _exit_stack, _session, _loop, _tools

    if not settings.MCP_ENABLED:
        print("[MCP] Disabled via settings (MCP_ENABLED=false)")
        return

    _loop = asyncio.get_running_loop()
    _exit_stack = AsyncExitStack()

    try:
        read_stream, write_stream = await _exit_stack.enter_async_context(
            sse_client(url=settings.MCP_SERVER_URL)
        )
        _session = await _exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

        await _session.initialize()
        tools_response = await _session.list_tools()
        _tools = [tool.name for tool in tools_response.tools]

        print(f"[MCP] Connected: {settings.MCP_SERVER_URL}")
        print(f"[MCP] Tools: {_tools}")
    except Exception as exc:
        _session = None
        _tools = []
        if _exit_stack is not None:
            await _exit_stack.aclose()
            _exit_stack = None
        print(f"[MCP] Not connected at startup: {str(exc)[:200]}")


async def shutdown_mcp_runtime() -> None:
    """Close MCP client resources on backend shutdown."""
    global _exit_stack, _session, _loop, _tools

    try:
        if _exit_stack is not None:
            await _exit_stack.aclose()
    finally:
        _exit_stack = None
        _session = None
        _loop = None
        _tools = []
        print("[MCP] Runtime stopped")


def has_live_session() -> bool:
    """Return true when backend has an active MCP session."""
    return _session is not None and _loop is not None


def get_tools() -> list:
    """Return tool names loaded at startup for debugging/visibility."""
    return list(_tools)


async def _call_tool_async(tool_name: str, arguments: dict):
    if _session is None:
        raise RuntimeError("MCP session is not active")
    return await _session.call_tool(tool_name, arguments)


def call_tool_threadsafe(tool_name: str, arguments: dict, timeout_seconds: int):
    """Call MCP tool from sync code by scheduling on backend event loop."""
    if _loop is None:
        raise RuntimeError("MCP loop is not initialized")

    future = asyncio.run_coroutine_threadsafe(
        _call_tool_async(tool_name, arguments),
        _loop,
    )
    return future.result(timeout=max(1, int(timeout_seconds)))

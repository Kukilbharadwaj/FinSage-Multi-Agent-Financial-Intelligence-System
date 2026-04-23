# mcp_bridge.py
# Shared MCP tool caller for synchronous agent usage.

import ast
import asyncio
import json
from typing import Any, Dict

from mcp import ClientSession
from mcp.client.sse import sse_client

from config.settings import settings
from mcp_runtime import has_live_session, call_tool_threadsafe


def is_mcp_enabled() -> bool:
    """Return whether MCP integration is enabled in app settings."""
    return bool(settings.MCP_ENABLED)


def _parse_tool_text(text: str) -> Any:
    """Parse MCP tool text into Python data when possible."""
    if not text:
        return {}

    # First try strict JSON.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback for Python dict-like strings from str(dict).
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    normalized = text.strip()
    if normalized.lower().startswith("error"):
        return {"error": normalized}
    return {"raw_text": normalized}


def _extract_mcp_result(result: Any) -> Any:
    """Extract text payload from MCP call_tool response."""
    if hasattr(result, "content") and isinstance(result.content, list):
        text_parts = []
        for item in result.content:
            if getattr(item, "type", "") == "text":
                text_parts.append(getattr(item, "text", ""))
        return _parse_tool_text("\n".join([part for part in text_parts if part]))
    return _parse_tool_text(str(result))


async def _call_mcp_tool_async(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Open a short-lived MCP session and invoke one tool."""
    timeout = max(1, int(settings.MCP_TIMEOUT_SECONDS))
    server_url = settings.MCP_SERVER_URL

    async def _invoke() -> Any:
        async with sse_client(url=server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return _extract_mcp_result(result)

    return await asyncio.wait_for(_invoke(), timeout=timeout)


def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Synchronous wrapper for MCP tool calls used by sync agents."""
    # Prefer backend-managed MCP client session started by main.py.
    if has_live_session():
        result = call_tool_threadsafe(
            tool_name=tool_name,
            arguments=arguments,
            timeout_seconds=settings.MCP_TIMEOUT_SECONDS,
        )
        return _extract_mcp_result(result)

    # Fallback path keeps app resilient if MCP restarts after backend startup.
    return asyncio.run(_call_mcp_tool_async(tool_name, arguments))

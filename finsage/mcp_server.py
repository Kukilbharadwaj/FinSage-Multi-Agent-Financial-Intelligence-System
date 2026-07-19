# mcp_server.py
# FinSage MCP server, defined with FastMCP.
#
# This module only DEFINES the server. Nothing here starts a process.
# The FastAPI backend talks to these tools over FastMCP's in-memory
# transport (see mcp_bridge.py), so there is no second server to run,
# no SSE port, and no network hop — an in-memory tool call measures
# ~2ms versus ~50ms+ over SSE.
#
# Running this file directly still exposes the same tools over stdio,
# which is useful for connecting an external MCP client (Claude Desktop,
# MCP Inspector) without changing any of the tool code.

import json
from typing import Any

from fastmcp import FastMCP

from tools.mf_tool import get_mf_details
from tools.news_tool import get_news
from tools.nse_tool import get_nse_option_chain, get_nse_quote
from tools.yahoo_tool import (
    get_company_profile,
    get_indian_market_status,
    get_intraday_data,
    get_stock_data,
)

mcp = FastMCP("FinSage")


def _ok(data: Any) -> str:
    """Serialise a tool result as JSON.

    The previous implementation returned str(dict), which produced Python
    repr with single quotes that json.loads could not parse — the bridge
    then fell back to ast.literal_eval or handed agents a {"raw_text": ...}
    blob. Emitting real JSON keeps the agent side simple and reliable.
    """
    return json.dumps(data, default=str, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"error": message[:300]})


@mcp.tool()
def nse_quote(symbol: str) -> str:
    """Fetch a live quote from NSE India — indices and individual stocks.

    Returns exchange-native price, change, day range, 52-week range, and for
    equities also sector, P/E, delivery percentage and volatility.

    Args:
        symbol: An index ("NIFTY 50", "BANKNIFTY") or a stock ("RELIANCE", "TCS").
    """
    try:
        return _ok(get_nse_quote(symbol))
    except Exception as exc:
        return _err(f"NSE quote failed for {symbol}: {exc}")


@mcp.tool()
def stock_data(symbol: str) -> str:
    """Fetch current stock or index price data from Yahoo Finance.

    Returns price, percent change, day high/low and the 52-week range.

    Args:
        symbol: The stock or index symbol, e.g. "RELIANCE" or "TCS".
    """
    try:
        return _ok(get_stock_data(symbol))
    except Exception as exc:
        return _err(f"Stock data failed for {symbol}: {exc}")


@mcp.tool()
def company_profile(symbol: str) -> str:
    """Fetch company fundamentals: sector, market cap, P/E, EPS, ROE, debt/equity.

    Args:
        symbol: The stock symbol, e.g. "TCS".
    """
    try:
        return _ok(get_company_profile(symbol))
    except Exception as exc:
        return _err(f"Company profile failed for {symbol}: {exc}")


@mcp.tool()
def intraday_data(symbol: str) -> str:
    """Fetch today's 5-minute intraday candles with VWAP and day high/low.

    Args:
        symbol: The stock or index symbol.
    """
    try:
        data = get_intraday_data(symbol)
        # Drop the raw candle arrays — agents only use the summary fields, and
        # ~75 candles of OHLCV would otherwise dominate the prompt.
        trimmed = {k: v for k, v in data.items() if k not in ("close", "high", "low", "volume", "dates")}
        trimmed["candle_count"] = len(data.get("close", []))
        return _ok(trimmed)
    except Exception as exc:
        return _err(f"Intraday data failed for {symbol}: {exc}")


@mcp.tool()
def options_chain(symbol: str, expiry: str = "") -> str:
    """Fetch a live NSE option chain with strikes, OI, IV, PCR and max pain.

    Sourced from NSE — Yahoo Finance carries no options data for Indian symbols.

    Args:
        symbol: NIFTY, BANKNIFTY, FINNIFTY, or an F&O stock symbol.
        expiry: Optional expiry as "DD-Mon-YYYY". Defaults to the nearest expiry.
    """
    try:
        return _ok(get_nse_option_chain(symbol, expiry=expiry or None))
    except Exception as exc:
        return _err(f"Option chain failed for {symbol}: {exc}")


@mcp.tool()
def market_status() -> str:
    """Return whether the Indian market is currently open, with IST session times."""
    try:
        return _ok(get_indian_market_status())
    except Exception as exc:
        return _err(f"Market status failed: {exc}")


@mcp.tool()
def mf_details(query: str) -> str:
    """Search a mutual fund and return its NAV, category and trailing returns.

    Args:
        query: Fund name or AMFI scheme code.
    """
    try:
        return _ok(get_mf_details(query))
    except Exception as exc:
        return _err(f"Mutual fund lookup failed for {query}: {exc}")


@mcp.tool()
def market_news(query: str = "Indian stock market", limit: int = 8) -> str:
    """Fetch recent Indian financial news headlines for a topic or symbol.

    Args:
        query: Topic or symbol to search, e.g. "RELIANCE" or "Indian stock market".
        limit: Maximum number of headlines to return.
    """
    try:
        return _ok(get_news(query, limit=limit))
    except Exception as exc:
        return _err(f"News fetch failed for {query}: {exc}")


if __name__ == "__main__":
    # Optional: expose the same tools to an external MCP client over stdio.
    # The FinSage backend does NOT need this — it uses the in-memory transport.
    mcp.run()

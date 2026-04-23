import asyncio
from typing import Dict, Any, List

from mcp.server.fastmcp import FastMCP

# Import FinSage tools
from tools.nse_tool import get_nse_quote
from tools.yahoo_tool import (
    get_stock_data,
    get_company_profile,
    get_intraday_data,
    get_options_chain,
)
from tools.mf_tool import get_mf_details

# Initialize FastMCP server on port 8001 (since FastAPI uses 8000)
mcp = FastMCP("FinSage", port=8001)

@mcp.tool()
def nse_quote(symbol: str) -> str:
    """
    Fetch live quote from NSE India.
    For indices use NIFTY, BANKNIFTY, SENSEX. For stocks use their symbol.
    
    Args:
        symbol: The stock or index symbol.
    """
    try:
        data = get_nse_quote(symbol)
        return str(data)
    except Exception as e:
        return f"Error fetching NSE quote: {str(e)}"


@mcp.tool()
def stock_data(symbol: str) -> str:
    """
    Fetch current stock/index data from Yahoo Finance.
    Returns price, change, high, low, and 52-week data.
    
    Args:
        symbol: The stock or index symbol (e.g., RELIANCE).
    """
    try:
        data = get_stock_data(symbol)
        return str(data)
    except Exception as e:
        return f"Error fetching Yahoo stock data: {str(e)}"


@mcp.tool()
def company_profile(symbol: str) -> str:
    """
    Fetch detailed company fundamentals from Yahoo Finance.
    Returns sector, industry, market cap, PE ratio, EPS, etc.
    
    Args:
        symbol: The stock symbol (e.g., TCS).
    """
    try:
        data = get_company_profile(symbol)
        return str(data)
    except Exception as e:
        return f"Error fetching company profile: {str(e)}"


@mcp.tool()
def intraday_data(symbol: str) -> str:
    """
    Fetch intraday data (5-minute candles for today) from Yahoo Finance.
    
    Args:
        symbol: The stock symbol.
    """
    try:
        data = get_intraday_data(symbol)
        # Convert list lengths or summarize to prevent huge output if necessary,
        # but for MCP tool calls, raw dict converted to string is fine.
        return str(data)
    except Exception as e:
        return f"Error fetching intraday data: {str(e)}"


@mcp.tool()
def options_chain(symbol: str) -> str:
    """
    Fetch options chain data from Yahoo Finance.
    Returns calls, puts, and PCR for the nearest expiry.
    
    Args:
        symbol: The stock symbol.
    """
    try:
        data = get_options_chain(symbol)
        return str(data)
    except Exception as e:
        return f"Error fetching options chain: {str(e)}"


@mcp.tool()
def mf_details(query: str) -> str:
    """
    Search for a mutual fund, get NAV + history, and calculate returns.
    
    Args:
        query: Name or AMFI code of the mutual fund.
    """
    try:
        data = get_mf_details(query)
        return str(data)
    except Exception as e:
        return f"Error fetching mutual fund details: {str(e)}"


def main():
    # Initialize and run the server using sse transport on port 8001
    print("Starting FinSage MCP Server on http://localhost:8001/sse")
    mcp.run(transport="sse")

if __name__ == "__main__":
    main()

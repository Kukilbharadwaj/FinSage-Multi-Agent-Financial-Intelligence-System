# agents/market_agent.py
# Fetches and summarizes live stock/index data WITH company fundamentals.
# Model: GROQ_STANDARD (llama-3.3-70b-versatile)

from datetime import datetime, timezone
from groq import Groq
from config.settings import settings
from config.models import GROQ_STANDARD
from mcp_bridge import call_mcp_tool, is_mcp_enabled
from tools.nse_tool import get_nse_quote
from tools.yahoo_tool import get_stock_data, get_company_profile


def run(state: dict) -> dict:
    """
    Fetch live market data AND company fundamentals for the detected stock/index.

    Tries NSE first for price, falls back to Yahoo Finance.
    Also fetches company profile for stock queries.
    """
    try:
        entities = state.get("entities", {})
        intent = state.get("intent", "")

        # Determine symbol to look up
        symbol = entities.get("stock") or entities.get("index") or "NIFTY 50"

        # Try MCP tools first (if enabled), then local tools as fallback.
        source = "NSE"
        try:
            if is_mcp_enabled():
                market_data = call_mcp_tool("nse_quote", {"symbol": symbol})
                source = "MCP:NSE"
            else:
                market_data = get_nse_quote(symbol)
        except Exception as nse_error:
            source = "Yahoo Finance"
            try:
                if is_mcp_enabled():
                    market_data = call_mcp_tool("stock_data", {"symbol": symbol})
                    source = "MCP:Yahoo"
                else:
                    market_data = get_stock_data(symbol)
            except Exception as yahoo_error:
                state["market_data"] = {
                    "symbol": symbol,
                    "error": f"NSE: {str(nse_error)[:80]} | Yahoo: {str(yahoo_error)[:80]}",
                    "source": "unavailable",
                }
                state["data_freshness"] = datetime.now(timezone.utc).isoformat()
                state["trace"].append(f"market_agent → {symbol} data fetch FAILED")
                return state

        state["market_data"] = market_data
        state["data_freshness"] = datetime.now(timezone.utc).isoformat()

        # Fetch company profile for stock queries (not index queries)
        profile = {}
        if intent == "stock":
            try:
                if is_mcp_enabled():
                    profile = call_mcp_tool("company_profile", {"symbol": symbol})
                else:
                    profile = get_company_profile(symbol)
                state["company_profile"] = profile
            except Exception:
                state["company_profile"] = {}

        # Generate summary using Groq
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            # Build richer prompt with fundamentals
            fundamentals_text = ""
            if profile:
                fundamentals_text = f"""
Company Fundamentals:
- Name: {profile.get('name', 'N/A')}
- Sector: {profile.get('sector', 'N/A')} | Industry: {profile.get('industry', 'N/A')}
- Market Cap: {profile.get('market_cap_formatted', 'N/A')}
- P/E Ratio: {profile.get('pe_ratio', 'N/A')} | Forward P/E: {profile.get('forward_pe', 'N/A')}
- EPS: ₹{profile.get('eps', 'N/A')}
- Dividend Yield: {profile.get('dividend_yield', 'N/A')}%
- Book Value: ₹{profile.get('book_value', 'N/A')} | P/B: {profile.get('price_to_book', 'N/A')}
- Revenue: {profile.get('revenue_formatted', 'N/A')}
- Profit Margin: {profile.get('profit_margin', 'N/A')}% | ROE: {profile.get('roe', 'N/A')}%
- Debt-to-Equity: {profile.get('debt_to_equity', 'N/A')}
- Beta: {profile.get('beta', 'N/A')}
- About: {profile.get('description', 'N/A')[:300]}"""

            summary_prompt = f"""Summarize the following market data and company profile in a comprehensive analysis.
Mention current price, change %, 52-week position, and if available: P/E valuation (is it overvalued/undervalued vs sector), 
dividend yield attractiveness, debt health, and growth prospects.
Be specific with numbers. Use ₹ symbol for Indian stocks.

Market Data:
- Symbol: {market_data.get('symbol', 'N/A')}
- Current Price: {market_data.get('price', 'N/A')}
- Change: {market_data.get('change', 'N/A')}%
- Day High: {market_data.get('high', 'N/A')}
- Day Low: {market_data.get('low', 'N/A')}
- 52-Week High: {market_data.get('52w_high', 'N/A')}
- 52-Week Low: {market_data.get('52w_low', 'N/A')}
- Data Source: {market_data.get('source', source)}
{fundamentals_text}"""

            response = client.chat.completions.create(
                model=GROQ_STANDARD,
                messages=[
                    {"role": "system", "content": "You are a concise Indian market data analyst specializing in both price action and fundamental analysis."},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            state["market_data"]["summary"] = response.choices[0].message.content.strip()

        except Exception as llm_error:
            # If LLM fails, create a simple summary from raw data
            state["market_data"]["summary"] = (
                f"{market_data.get('symbol', 'N/A')} is trading at ₹{market_data.get('price', 'N/A')} "
                f"({market_data.get('change', 'N/A')}% change). "
                f"52-week range: ₹{market_data.get('52w_low', 'N/A')} - ₹{market_data.get('52w_high', 'N/A')}."
            )

        state["trace"].append(f"market_agent → {symbol} fetched from {source}" + (" + fundamentals" if profile else ""))

    except Exception as e:
        state["market_data"] = {"error": str(e)[:200]}
        state["data_freshness"] = datetime.now(timezone.utc).isoformat()
        state["trace"].append(f"market_agent → ERROR: {str(e)[:100]}")

    return state

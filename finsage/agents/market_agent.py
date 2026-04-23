# agents/market_agent.py
# Fetches and summarizes live stock/index data.
# Model: GROQ_STANDARD (llama-3.3-70b-versatile)

from datetime import datetime, timezone
from groq import Groq
from config.settings import settings
from config.models import GROQ_STANDARD
from tools.nse_tool import get_nse_quote
from tools.yahoo_tool import get_stock_data


def run(state: dict) -> dict:
    """
    Fetch live market data for the detected stock/index and generate a summary.

    Tries NSE first, falls back to Yahoo Finance.
    Stores raw data in state["market_data"] and summary in state["market_data"]["summary"].
    """
    try:
        entities = state.get("entities", {})

        # Determine symbol to look up
        symbol = entities.get("stock") or entities.get("index") or "NIFTY 50"

        # Try NSE first, fall back to Yahoo Finance
        source = "NSE"
        try:
            market_data = get_nse_quote(symbol)
        except Exception as nse_error:
            source = "Yahoo Finance"
            try:
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

        # Generate summary using Groq
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            summary_prompt = f"""Summarize the following market data in 3-4 sentences. 
Mention the current price, whether it's up or down (change %), and how it compares to 52-week high/low.
Be specific with numbers. Use ₹ symbol for Indian stocks.

Market Data:
- Symbol: {market_data.get('symbol', 'N/A')}
- Current Price: {market_data.get('price', 'N/A')}
- Change: {market_data.get('change', 'N/A')}%
- Day High: {market_data.get('high', 'N/A')}
- Day Low: {market_data.get('low', 'N/A')}
- 52-Week High: {market_data.get('52w_high', 'N/A')}
- 52-Week Low: {market_data.get('52w_low', 'N/A')}
- Data Source: {market_data.get('source', source)}"""

            response = client.chat.completions.create(
                model=GROQ_STANDARD,
                messages=[
                    {"role": "system", "content": "You are a concise Indian market data analyst."},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )

            state["market_data"]["summary"] = response.choices[0].message.content.strip()

        except Exception as llm_error:
            # If LLM fails, create a simple summary from raw data
            state["market_data"]["summary"] = (
                f"{market_data.get('symbol', 'N/A')} is trading at ₹{market_data.get('price', 'N/A')} "
                f"({market_data.get('change', 'N/A')}% change). "
                f"52-week range: ₹{market_data.get('52w_low', 'N/A')} - ₹{market_data.get('52w_high', 'N/A')}."
            )

        state["trace"].append(f"market_agent → {symbol} fetched from {source}")

    except Exception as e:
        state["market_data"] = {"error": str(e)[:200]}
        state["data_freshness"] = datetime.now(timezone.utc).isoformat()
        state["trace"].append(f"market_agent → ERROR: {str(e)[:100]}")

    return state

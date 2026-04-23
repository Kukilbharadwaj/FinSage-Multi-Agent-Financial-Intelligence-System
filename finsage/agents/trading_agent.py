# agents/trading_agent.py
# Intraday and Options/F&O trading analysis agent.
# Fetches live intraday data, options chain, and provides trading signals.
# Model: GROQ_REASONING (qwen/qwen3-32b) — step-by-step trading strategy

from groq import Groq
from config.settings import settings
from config.models import GROQ_REASONING
from mcp_bridge import call_mcp_tool, is_mcp_enabled
from tools.yahoo_tool import get_intraday_data, get_options_chain, get_indian_market_status
from tools.technical_tool import calculate_indicators
from rag.knowledge_base import query_kb


def run(state: dict) -> dict:
    """
    Analyze intraday and F&O trading opportunities using live data.

    Fetches intraday candles, options chain data, and uses RAG context
    for trading rules. Produces actionable trading signals.
    """
    try:
        entities = state.get("entities", {})
        raw_query = state.get("raw_query", "").lower()

        # Determine symbol
        symbol = entities.get("stock") or entities.get("index") or "NIFTY 50"

        # Get trading rules from RAG
        rag_context = query_kb("intraday trading options F&O strategies risk management stop-loss")
        state["rag_context"] = rag_context

        # Fetch intraday data
        intraday = {}
        try:
            if is_mcp_enabled():
                intraday = call_mcp_tool("intraday_data", {"symbol": symbol})
            else:
                intraday = get_intraday_data(symbol)
            state["intraday_data"] = intraday
        except Exception as e:
            state["intraday_data"] = {"error": str(e)[:100]}

        # Fetch options chain if query is about options/F&O
        options_data = {}
        is_options_query = any(kw in raw_query for kw in [
            "option", "call", "put", "f&o", "strike", "premium",
            "straddle", "strangle", "iron condor", "nifty option",
            "banknifty option", "pcr", "max pain", "oi", "open interest"
        ])

        if is_options_query:
            try:
                if is_mcp_enabled():
                    options_data = call_mcp_tool("options_chain", {"symbol": symbol})
                else:
                    options_data = get_options_chain(symbol)
                state["options_chain"] = options_data
            except Exception as e:
                state["options_chain"] = {"error": str(e)[:100]}

        market_status = (intraday or {}).get("market_status") or get_indian_market_status()
        is_market_open = bool(market_status.get("is_open"))

        # Build analysis prompt
        system_message = """You are an expert Indian intraday and options analyst.
    Write like a human mentor, not a bot.

    Rules:
    1) First decide whether market is OPEN or CLOSED from provided status.
    2) If market is CLOSED, do NOT give immediate "buy now" or "sell now" calls. Provide a next-session plan with trigger levels.
    3) If market is OPEN, provide action-ready guidance with trigger-based entry, stop-loss, and fast target windows.
    4) You may give short-horizon guidance (for example 5-15 minute window), but avoid guaranteed claims.
    5) Use plain, conversational language with concrete prices in ₹.
    6) Always include risk guardrails and mention this is educational only."""

        # Build data sections
        data_parts = []

        if intraday and "error" not in intraday:
            data_parts.append(f"""INTRADAY DATA for {symbol}:
- Last Price: ₹{intraday.get('last_price', 'N/A')}
- Day High: ₹{intraday.get('day_high', 'N/A')}
- Day Low: ₹{intraday.get('day_low', 'N/A')}
- VWAP: ₹{intraday.get('vwap', 'N/A')}
    - Total Volume: {intraday.get('total_volume', 'N/A')}
    - Last Candle Time (IST): {intraday.get('last_candle_time_ist', 'N/A')}
    - Candle Delay: {intraday.get('candle_delay_minutes', 'N/A')} minutes
    - Live Data: {intraday.get('is_live_data', False)}""")

        data_parts.append(f"""MARKET STATUS (IST):
    - Status: {market_status.get('status', 'unknown').upper()}
    - Is Open: {is_market_open}
    - Current Time: {market_status.get('current_time_ist', 'N/A')}
    - Next Open: {market_status.get('next_open_ist', 'N/A')}
    - Today Close: {market_status.get('today_close_ist', 'N/A')}
    - Note: {market_status.get('reason', 'N/A')}""")

        if options_data and "error" not in options_data:
            # Format top calls and puts
            top_calls = options_data.get("calls", [])[:5]
            top_puts = options_data.get("puts", [])[:5]

            calls_text = "\n".join([
                f"  Strike ₹{c['strike']}: Premium ₹{c['lastPrice']}, OI={c['openInterest']}, IV={c['impliedVolatility']}%"
                for c in top_calls
            ])
            puts_text = "\n".join([
                f"  Strike ₹{p['strike']}: Premium ₹{p['lastPrice']}, OI={p['openInterest']}, IV={p['impliedVolatility']}%"
                for p in top_puts
            ])

            data_parts.append(f"""OPTIONS CHAIN for {symbol} (Expiry: {options_data.get('expiry', 'N/A')}):
- Current Price: ₹{options_data.get('current_price', 'N/A')}
- Put-Call Ratio (PCR): {options_data.get('pcr', 'N/A')}
- Max Pain: ₹{options_data.get('max_pain', 'N/A')}
- Total Call OI: {options_data.get('total_call_oi', 'N/A')}
- Total Put OI: {options_data.get('total_put_oi', 'N/A')}

Top Calls (by volume):
{calls_text}

Top Puts (by volume):
{puts_text}""")

        data_parts.append(f"""TRADING RULES (from knowledge base):
{rag_context[:600]}""")

        data_string = "\n\n".join(data_parts)

        user_message = f"""User's question: "{state['raw_query']}"

{data_string}

    Provide a detailed trading analysis in this style:

    1. Market Status Summary (open/closed + what that means right now)
    2. What I would do now (human style)
    3. Key levels (support/resistance/VWAP)
    4. {"Options plan with strikes, triggers, invalidation" if is_options_query else "Intraday plan with trigger, stop-loss, target and time window"}
    5. Quick scenario plan: if momentum continues vs if reversal starts
    6. Risk rules: max loss, no revenge trade, position sizing
    7. ⚠️ Warning: include SEBI's data that 9/10 F&O traders lose money

    If market is CLOSED:
    - Give "next session preparation" and "opening 15-min observation plan".
    - No immediate execution command.

    If market is OPEN:
    - You can give actionable setup like "Buy only above X" / "Sell below Y".
    - Time window can be short (for example 5-15 min), but no guaranteed return claims.

Be specific with prices in ₹. This is for educational purposes only."""

        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            response = client.chat.completions.create(
                model=GROQ_REASONING,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.5,
                max_tokens=2000,
                reasoning_format="hidden",
            )

            state["trading_analysis"] = response.choices[0].message.content.strip()

        except Exception as llm_error:
            state["trading_analysis"] = (
                f"Trading analysis could not be completed: {str(llm_error)[:100]}. "
                "Please use a professional trading terminal for real-time analysis."
            )

        state["trace"].append(
            f"trading_agent → {symbol} "
            f"{'options + intraday' if is_options_query else 'intraday'} analysis ({market_status.get('status', 'unknown')})"
        )

    except Exception as e:
        state["trading_analysis"] = f"Trading agent error: {str(e)[:200]}"
        state["trace"].append(f"trading_agent → ERROR: {str(e)[:100]}")

    return state

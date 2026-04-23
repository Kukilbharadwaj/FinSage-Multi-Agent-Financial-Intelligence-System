# agents/trading_agent.py
# Intraday and Options/F&O trading analysis agent.
# Fetches live intraday data, options chain, and provides trading signals.
# Model: GROQ_REASONING (qwen/qwen3-32b) — step-by-step trading strategy

from groq import Groq
from config.settings import settings
from config.models import GROQ_REASONING
from tools.yahoo_tool import get_intraday_data, get_options_chain
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
                options_data = get_options_chain(symbol)
                state["options_chain"] = options_data
            except Exception as e:
                state["options_chain"] = {"error": str(e)[:100]}

        # Build analysis prompt
        system_message = """You are an expert Indian stock market trader specializing in intraday and F&O trading.
Analyze the data and provide clear, actionable trading signals with specific entry, target, and stop-loss prices.
Always emphasize risk management. Use ₹ symbol for prices."""

        # Build data sections
        data_parts = []

        if intraday and "error" not in intraday:
            data_parts.append(f"""INTRADAY DATA for {symbol}:
- Last Price: ₹{intraday.get('last_price', 'N/A')}
- Day High: ₹{intraday.get('day_high', 'N/A')}
- Day Low: ₹{intraday.get('day_low', 'N/A')}
- VWAP: ₹{intraday.get('vwap', 'N/A')}
- Total Volume: {intraday.get('total_volume', 'N/A')}""")

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

Provide a detailed trading analysis:

1. **Market Outlook**: Current trend direction and strength based on intraday data
2. **Key Levels**: Support, resistance, VWAP position
{"3. **Options Analysis**: PCR interpretation, max pain implication, recommended option strategy" if is_options_query else "3. **Intraday Strategy**: Specific entry/exit levels"}
4. **Trade Setup**: Specific entry price, target price, stop-loss price
5. **Risk Management**: Position sizing, max risk per trade
6. **⚠️ Warning**: Include SEBI's data that 9/10 F&O traders lose money

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
            f"{'options + intraday' if is_options_query else 'intraday'} analysis"
        )

    except Exception as e:
        state["trading_analysis"] = f"Trading agent error: {str(e)[:200]}"
        state["trace"].append(f"trading_agent → ERROR: {str(e)[:100]}")

    return state

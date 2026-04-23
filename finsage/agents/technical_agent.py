# agents/technical_agent.py
# Calculates and interprets technical indicators.
# Model: GROQ_REASONING (qwen/qwen3-32b) — step-by-step reasoning for TA interpretation

from groq import Groq
from config.settings import settings
from config.models import GROQ_REASONING
from tools.yahoo_tool import get_ohlcv
from tools.technical_tool import calculate_indicators

# Map common index names to Yahoo symbols for OHLCV data
INDEX_SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN",
}


def run(state: dict) -> dict:
    """
    Fetch OHLCV data, calculate technical indicators, and use qwq-32b
    to reason through the trading signal step by step.

    Stores results in state["technical_signals"] and state["ohlcv"].
    """
    try:
        entities = state.get("entities", {})
        intent = state.get("intent", "")

        # Determine symbol
        symbol = entities.get("stock") or entities.get("index") or "NIFTY 50"

        # For index queries, map to Yahoo symbol
        symbol_upper = symbol.upper().strip()
        if symbol_upper in INDEX_SYMBOL_MAP:
            yahoo_symbol = INDEX_SYMBOL_MAP[symbol_upper]
        else:
            yahoo_symbol = symbol

        # Fetch OHLCV data
        try:
            ohlcv = get_ohlcv(yahoo_symbol, period="3mo")
            state["ohlcv"] = ohlcv
        except Exception as ohlcv_error:
            state["ohlcv"] = {}
            state["technical_signals"] = {"error": f"Could not fetch OHLCV: {str(ohlcv_error)[:100]}"}
            state["trace"].append(f"technical_agent → OHLCV fetch failed for {symbol}")
            return state

        if not ohlcv or not ohlcv.get("close"):
            state["technical_signals"] = {"error": "Insufficient OHLCV data"}
            state["trace"].append(f"technical_agent → insufficient data for {symbol}")
            return state

        # Calculate indicators
        indicators = calculate_indicators(ohlcv)

        if not indicators:
            state["technical_signals"] = {"error": "Indicator calculation failed — insufficient data points"}
            state["trace"].append(f"technical_agent → indicator calculation failed for {symbol}")
            return state

        # Build messages for qwen3-32b reasoning
        system_message = "You are a technical analysis expert for Indian stock markets. Analyze indicators step by step and give clear trading signals."

        user_message = f"""I have the following technical indicators for {symbol}:

- Current Price: ₹{indicators['current_price']}
- EMA 20 (20-day Exponential Moving Average): ₹{indicators['ema20']}
- EMA 50 (50-day Exponential Moving Average): ₹{indicators['ema50']}
- RSI (14-day Relative Strength Index): {indicators['rsi']}
- MACD Histogram: {indicators['macd']}
- Support Level (20-day low): ₹{indicators['support']}
- Resistance Level (20-day high): ₹{indicators['resistance']}
- Trend (EMA crossover): {indicators['trend']}

Please analyze these indicators step by step:

Step 1: Interpret the trend from EMA crossover. Is EMA20 above or below EMA50? What does this mean for the short-term vs medium-term trend?

Step 2: Interpret the RSI value. Is it overbought (above 70), oversold (below 30), or neutral? What does this suggest about momentum?

Step 3: Identify the support and resistance levels. How far is the current price from support and resistance? Are these levels significant?

Step 4: Give a clear trading signal: BUY, SELL, or HOLD. Provide a specific reason based on the confluence of indicators above.

Step 5: Suggest a stop-loss price level based on the support level. Calculate the risk percentage from current price to stop-loss.

Be specific with numbers. Use ₹ symbol for prices. Keep your analysis practical for Indian retail investors."""

        # Call qwen3-32b reasoning model
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            response = client.chat.completions.create(
                model=GROQ_REASONING,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.6,
                max_tokens=1500,
                reasoning_format="hidden",
            )

            analysis = response.choices[0].message.content.strip()

            # Store both raw indicators and LLM analysis
            state["technical_signals"] = {
                **indicators,
                "analysis": analysis,
            }

        except Exception as llm_error:
            # If LLM fails, store raw indicators without analysis
            state["technical_signals"] = {
                **indicators,
                "analysis": f"Technical analysis interpretation unavailable: {str(llm_error)[:100]}",
            }

        trend = indicators.get("trend", "unknown")
        state["trace"].append(f"technical_agent → {trend} signal")

    except Exception as e:
        state["technical_signals"] = {"error": str(e)[:200]}
        state["trace"].append(f"technical_agent → ERROR: {str(e)[:100]}")

    return state

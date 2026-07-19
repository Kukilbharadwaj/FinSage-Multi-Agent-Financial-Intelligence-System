# agents/technical_agent.py
# Calculates and interprets technical indicators.
# Model: GROQ_REASONING (openai/gpt-oss-120b) — step-by-step reasoning for TA interpretation
#
# Stage 3 agent — no upstream dependencies (pure calculation + LLM interpretation).
# Does NOT use RAG.
# Writes: state["technical_analysis"]

from llm import Groq
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

    Writes state["technical_analysis"] with structured output:
        - signals: dict of calculated indicators (ema20, ema50, rsi, macd, etc.)
        - ohlcv: raw OHLCV data
        - analysis: LLM interpretation text
    """
    try:
        entities = state.get("entities", {})

        # Determine symbol
        symbol = entities.get("stock") or entities.get("index") or "NIFTY 50"

        # For index queries, map to Yahoo symbol
        symbol_upper = symbol.upper().strip()
        if symbol_upper in INDEX_SYMBOL_MAP:
            yahoo_symbol = INDEX_SYMBOL_MAP[symbol_upper]
        else:
            yahoo_symbol = symbol

        # Fetch OHLCV data
        ohlcv = {}
        try:
            ohlcv = get_ohlcv(yahoo_symbol, period="3mo")
        except Exception as ohlcv_error:
            state["technical_analysis"] = {
                "signals": {},
                "ohlcv": {},
                "analysis": f"Could not fetch OHLCV: {str(ohlcv_error)[:100]}",
            }
            state["trace"].append(f"technical_agent → OHLCV fetch failed for {symbol}")
            return state

        if not ohlcv or not ohlcv.get("close"):
            state["technical_analysis"] = {
                "signals": {},
                "ohlcv": {},
                "analysis": "Insufficient OHLCV data for technical analysis",
            }
            state["trace"].append(f"technical_agent → insufficient data for {symbol}")
            return state

        # Calculate indicators
        indicators = calculate_indicators(ohlcv)

        if not indicators:
            state["technical_analysis"] = {
                "signals": {},
                "ohlcv": ohlcv,
                "analysis": "Indicator calculation failed — insufficient data points",
            }
            state["trace"].append(f"technical_agent → indicator calculation failed for {symbol}")
            return state

        # Build messages for the reasoning model
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

        analysis_text = ""

        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            response = client.chat.completions.create(
                name="technical_llm",
                model=GROQ_REASONING,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.6,
                # See mutual_fund_agent: hidden reasoning ate the budget before
                # emitting visible text. Synthesis reads 700 chars of this one.
                max_tokens=1200,
                reasoning_format="hidden",
                reasoning_effort="low",
            )

            analysis_text = response.choices[0].message.content.strip()

        except Exception as llm_error:
            analysis_text = f"Technical analysis interpretation unavailable: {str(llm_error)[:100]}"

        # ── Write structured output to communication bus ──
        state["technical_analysis"] = {
            "signals": indicators,
            "ohlcv": ohlcv,
            "analysis": analysis_text,
        }

        trend = indicators.get("trend", "unknown")
        state["trace"].append(f"technical_agent → {trend} signal for {symbol}")

    except Exception as e:
        state["technical_analysis"] = {
            "signals": {},
            "ohlcv": {},
            "analysis": f"Technical agent error: {str(e)[:200]}",
        }
        state["trace"].append(f"technical_agent → ERROR: {str(e)[:100]}")

    return state

# tools/technical_tool.py
# Pure mathematical calculations on OHLCV data.
# No API calls, no LLM calls. Uses the `ta` library and pandas.

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator


def calculate_indicators(ohlcv: dict) -> dict:
    """
    Calculate technical indicators from OHLCV data.

    Args:
        ohlcv: Dict with keys: close, high, low, volume (lists from get_ohlcv)

    Returns:
        Dict with: current_price, ema20, ema50, rsi, macd, support, resistance, trend
        All values rounded to 2 decimal places.
        Returns empty dict if data is insufficient.
    """
    if not ohlcv or not ohlcv.get("close") or len(ohlcv["close"]) < 50:
        return {}

    try:
        close = pd.Series(ohlcv["close"])
        high = pd.Series(ohlcv["high"])
        low = pd.Series(ohlcv["low"])

        # EMA 20 and EMA 50
        ema20_indicator = EMAIndicator(close=close, window=20)
        ema50_indicator = EMAIndicator(close=close, window=50)
        ema20 = ema20_indicator.ema_indicator().iloc[-1]
        ema50 = ema50_indicator.ema_indicator().iloc[-1]

        # RSI with window 14
        rsi_indicator = RSIIndicator(close=close, window=14)
        rsi = rsi_indicator.rsi().iloc[-1]

        # MACD diff
        macd_indicator = MACD(close=close)
        macd_diff = macd_indicator.macd_diff().iloc[-1]

        # Support: minimum of lows over last 20 candles
        support = low.iloc[-20:].min()

        # Resistance: maximum of highs over last 20 candles
        resistance = high.iloc[-20:].max()

        # Trend determination
        if ema20 > ema50:
            trend = "bullish"
        elif ema20 < ema50:
            trend = "bearish"
        else:
            trend = "sideways"

        current_price = close.iloc[-1]

        return {
            "current_price": round(float(current_price), 2),
            "ema20": round(float(ema20), 2),
            "ema50": round(float(ema50), 2),
            "rsi": round(float(rsi), 2),
            "macd": round(float(macd_diff), 2),
            "support": round(float(support), 2),
            "resistance": round(float(resistance), 2),
            "trend": trend,
        }

    except Exception:
        return {}

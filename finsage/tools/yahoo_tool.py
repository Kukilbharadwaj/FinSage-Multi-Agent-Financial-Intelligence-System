# tools/yahoo_tool.py
# yfinance wrapper for stock data and historical OHLCV.
# This is the fallback when NSE scraper fails, and the primary source for historical data.

import yfinance as yf
from typing import Optional


# Maps common Indian stock/index names to Yahoo Finance symbols
SYMBOL_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "WIPRO": "WIPRO.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "MARUTI": "MARUTI.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN",
}


def _resolve_symbol(symbol: str) -> str:
    """Convert common Indian stock names to Yahoo Finance symbols."""
    symbol = symbol.upper().strip()

    # Check the mapping first
    if symbol in SYMBOL_MAP:
        return SYMBOL_MAP[symbol]

    # If symbol already ends with .NS or starts with ^, use as-is
    if symbol.endswith(".NS") or symbol.endswith(".BO") or symbol.startswith("^"):
        return symbol

    # Default: append .NS for NSE listing
    return f"{symbol}.NS"


def get_stock_data(symbol: str) -> dict:
    """
    Fetch current stock/index data from Yahoo Finance.

    Returns dict with: symbol, price, change, high, low, 52w_high, 52w_low, source
    """
    try:
        yahoo_symbol = _resolve_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)
        info = ticker.info

        # Get current price — try multiple fields
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)

        # Get 52-week data
        high_52w = info.get("fiftyTwoWeekHigh", 0)
        low_52w = info.get("fiftyTwoWeekLow", 0)

        # Get day range
        day_high = info.get("dayHigh") or info.get("regularMarketDayHigh", 0)
        day_low = info.get("dayLow") or info.get("regularMarketDayLow", 0)

        # Calculate change percentage
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose", 0)
        if prev_close and prev_close > 0:
            change_pct = round(((price - prev_close) / prev_close) * 100, 2)
        else:
            change_pct = 0.0

        return {
            "symbol": symbol.upper(),
            "price": round(float(price), 2),
            "change": change_pct,
            "high": round(float(day_high), 2) if day_high else 0,
            "low": round(float(day_low), 2) if day_low else 0,
            "52w_high": round(float(high_52w), 2) if high_52w else 0,
            "52w_low": round(float(low_52w), 2) if low_52w else 0,
            "source": "Yahoo Finance",
        }

    except Exception as e:
        raise Exception(f"Yahoo Finance error for '{symbol}': {str(e)}")


def get_ohlcv(symbol: str, period: str = "3mo") -> dict:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Returns dict with: close, high, low, volume (as lists), dates (as list of strings).
    Used by the technical agent for indicator calculations.
    """
    try:
        yahoo_symbol = _resolve_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period=period, interval="1d")

        if df.empty:
            return {}

        return {
            "close": df["Close"].tolist(),
            "high": df["High"].tolist(),
            "low": df["Low"].tolist(),
            "volume": df["Volume"].tolist(),
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        }

    except Exception as e:
        raise Exception(f"Yahoo Finance OHLCV error for '{symbol}': {str(e)}")

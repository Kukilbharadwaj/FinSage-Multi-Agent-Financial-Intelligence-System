# tools/nse_tool.py
# Scrapes NSE India for live stock and index quotes.
# NSE blocks bots, so we handle session cookies correctly.

import requests
from typing import Optional


def _create_nse_session() -> requests.Session:
    """Create a requests session with proper headers and cookies for NSE."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com",
    })
    # Hit homepage first to get session cookies — required by NSE
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass  # Continue even if homepage fails; data request will fail gracefully
    return session


def get_nse_quote(symbol: str) -> dict:
    """
    Fetch live quote from NSE India.

    For indices (NIFTY, NIFTY 50, BANKNIFTY): uses allIndices API.
    For stocks: uses quote-equity API.

    Returns dict with: symbol, price, change, high, low, 52w_high, 52w_low, source
    Raises Exception if NSE is unreachable or data not found.
    """
    symbol = symbol.upper().strip()
    session = _create_nse_session()

    # Check if this is an index query
    index_names = {
        "NIFTY": "NIFTY 50",
        "NIFTY 50": "NIFTY 50",
        "NIFTY50": "NIFTY 50",
        "BANKNIFTY": "NIFTY BANK",
        "NIFTY BANK": "NIFTY BANK",
        "SENSEX": "SENSEX",
    }

    if symbol in index_names:
        return _get_index_quote(session, symbol, index_names[symbol])
    else:
        return _get_equity_quote(session, symbol)


def _get_index_quote(session: requests.Session, original_symbol: str, nse_name: str) -> dict:
    """Fetch index data from NSE allIndices API."""
    try:
        url = "https://www.nseindia.com/api/allIndices"
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        for index_data in data.get("data", []):
            if index_data.get("index", "").upper() == nse_name.upper():
                return {
                    "symbol": original_symbol,
                    "price": float(index_data.get("last", 0)),
                    "change": float(index_data.get("percentChange", 0)),
                    "high": float(index_data.get("high", 0)),
                    "low": float(index_data.get("low", 0)),
                    "52w_high": float(index_data.get("yearHigh", 0)),
                    "52w_low": float(index_data.get("yearLow", 0)),
                    "source": "NSE",
                }

        raise Exception(f"Index '{original_symbol}' not found in NSE data")

    except requests.exceptions.RequestException as e:
        raise Exception(f"NSE API unreachable for index '{original_symbol}': {str(e)}")


def _get_equity_quote(session: requests.Session, symbol: str) -> dict:
    """Fetch equity quote from NSE quote-equity API."""
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        price_info = data.get("priceInfo", {})
        week52 = price_info.get("weekHighLow", {})

        return {
            "symbol": symbol,
            "price": float(price_info.get("lastPrice", 0)),
            "change": float(price_info.get("pChange", 0)),
            "high": float(price_info.get("intraDayHighLow", {}).get("max", 0)),
            "low": float(price_info.get("intraDayHighLow", {}).get("min", 0)),
            "52w_high": float(week52.get("max", 0)),
            "52w_low": float(week52.get("min", 0)),
            "source": "NSE",
        }

    except requests.exceptions.RequestException as e:
        raise Exception(f"NSE API unreachable for equity '{symbol}': {str(e)}")
    except (KeyError, ValueError, TypeError) as e:
        raise Exception(f"Failed to parse NSE data for '{symbol}': {str(e)}")

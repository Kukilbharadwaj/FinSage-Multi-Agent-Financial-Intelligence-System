# tools/nse_tool.py
# NSE India live data: index quotes, equity quotes, and the F&O option chain.
#
# Two things are required to talk to NSE at all:
#
#   1. BROWSER TLS FINGERPRINT. NSE sits behind Akamai, which fingerprints the
#      TLS handshake. Plain `requests` gets 403 on the homepage itself. curl_cffi
#      with impersonate="chrome" presents a real Chrome fingerprint and gets 200.
#      (This is the same library yfinance uses, and it is already a dependency.)
#
#   2. THE CURRENT ENDPOINTS. NSE migrated its APIs and the old paths are gone:
#        /api/quote-equity          -> 403, permanently blocked at the WAF
#        /api/equity-stockIndices   -> 404, retired
#        /api/option-chain-indices  -> 404, retired
#      The live equivalents, read from the site's own JS bundle, are:
#        /api/allIndices                                    -> index quotes
#        /api/NextApi/apiClient/GetQuoteApi?functionName=... -> equity quotes
#        /api/option-chain-v3        (REQUIRES &expiry=)     -> option chain
#        /api/option-chain-contract-info                     -> expiry list
#
# The equity call needs marketType=N (normal market); "equity"/"CM" return 404.
#
# The warmed session is cached at module level — previously every single call
# re-hit the NSE homepage with a 10s timeout before doing any real work.

import threading
import time
from typing import Optional

import requests

try:
    from curl_cffi import requests as curl_requests

    _HAS_CURL_CFFI = True
except ImportError:  # pragma: no cover - falls back to plain requests
    curl_requests = None
    _HAS_CURL_CFFI = False

# Index name → the exact label used in the allIndices payload
INDEX_NAMES = {
    "NIFTY": "NIFTY 50",
    "NIFTY 50": "NIFTY 50",
    "NIFTY50": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTY BANK": "NIFTY BANK",
    "FINNIFTY": "NIFTY FINANCIAL SERVICES",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
    "SENSEX": "SENSEX",
}

# Symbols NSE exposes an index option chain for
INDEX_OPTION_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}

_SESSION: Optional[requests.Session] = None
_SESSION_CREATED_AT: float = 0.0
_SESSION_TTL_SECONDS = 600  # re-warm every 10 minutes
_SESSION_LOCK = threading.Lock()


def _new_session():
    """Create a browser-impersonating session, falling back to plain requests."""
    if _HAS_CURL_CFFI:
        # impersonate="chrome" is what gets us past Akamai's TLS fingerprinting.
        return curl_requests.Session(impersonate="chrome")

    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)
    return session


def _get_session(referer: str = "https://www.nseindia.com"):
    """Return a cached, cookie-warmed NSE session, re-warming it on TTL expiry."""
    global _SESSION, _SESSION_CREATED_AT

    with _SESSION_LOCK:
        fresh = _SESSION is not None and (time.time() - _SESSION_CREATED_AT) < _SESSION_TTL_SECONDS
        if not fresh:
            session = _new_session()
            # Warm-up mints the Akamai cookies (_abck, bm_sz) the API layer wants.
            try:
                session.get("https://www.nseindia.com", timeout=10)
            except Exception:
                pass
            _SESSION = session
            _SESSION_CREATED_AT = time.time()

        _SESSION.headers["Referer"] = referer
        return _SESSION


def _reset_session() -> None:
    """Force the next call to build a brand new session."""
    global _SESSION, _SESSION_CREATED_AT
    with _SESSION_LOCK:
        _SESSION = None
        _SESSION_CREATED_AT = 0.0


def _get_json(url: str, referer: str = "https://www.nseindia.com", timeout: int = 10) -> dict:
    """GET a JSON endpoint, retrying once with a fresh session on failure."""
    last_error = ""
    for attempt in (1, 2):
        session = _get_session(referer)
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = str(exc)[:120]
            if attempt == 2:
                raise Exception(f"NSE request failed for {url}: {last_error}")
            _reset_session()  # cookies likely went stale — rebuild and retry
    return {}


def get_nse_quote(symbol: str) -> dict:
    """
    Fetch a live quote from NSE — indices and equities both supported.

    Returns dict with: symbol, price, change, high, low, 52w_high, 52w_low, source
    """
    symbol = symbol.upper().strip()

    if symbol in INDEX_NAMES:
        return _get_index_quote(symbol, INDEX_NAMES[symbol])

    return get_nse_equity_quote(symbol)


def _get_index_quote(original_symbol: str, nse_name: str) -> dict:
    """Fetch index data from the NSE allIndices API."""
    data = _get_json("https://www.nseindia.com/api/allIndices")

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
                "pe": index_data.get("pe"),
                "pb": index_data.get("pb"),
                "source": "NSE",
            }

    raise Exception(f"Index '{original_symbol}' not found in NSE allIndices response")


_QUOTE_API = "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"


def _equity_referer(symbol: str) -> str:
    return f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"


def get_nse_equity_quote(symbol: str) -> dict:
    """
    Fetch a live EQUITY quote from NSE.

    Uses the current NextApi endpoint. The legacy /api/quote-equity path is
    permanently WAF-blocked (403) and cannot be used regardless of headers.

    Returns dict with: symbol, price, change, high, low, open, prev_close,
    52w_high, 52w_low, volume, vwap, market_cap, pe_ratio, sector, source
    """
    symbol = symbol.upper().strip()
    referer = _equity_referer(symbol)

    # marketType=N is required — "equity", "CM" and "normal" all return 404.
    data = _get_json(
        f"{_QUOTE_API}?functionName=getSymbolData"
        f"&marketType=N&series=EQ&symbol={symbol}",
        referer=referer,
        timeout=12,
    )

    responses = data.get("equityResponse") or []
    if not responses:
        raise Exception(f"NSE returned no equity data for '{symbol}'")

    payload = responses[0]
    meta = payload.get("metaData") or {}
    price_info = payload.get("priceInfo") or {}
    trade_info = payload.get("tradeInfo") or {}
    sec_info = payload.get("secInfo") or {}

    # During market hours lastPrice is live; after close it settles to closePrice.
    last_price = trade_info.get("lastPrice") or meta.get("closePrice") or meta.get("previousClose") or 0

    return {
        "symbol": symbol,
        "company_name": meta.get("companyName", ""),
        "price": float(last_price or 0),
        "change": float(meta.get("pChange") or 0),
        "change_abs": float(meta.get("change") or 0),
        "open": float(meta.get("open") or 0),
        "high": float(meta.get("dayHigh") or 0),
        "low": float(meta.get("dayLow") or 0),
        "prev_close": float(meta.get("previousClose") or 0),
        "vwap": float(meta.get("averagePrice") or 0),
        "52w_high": float(price_info.get("yearHigh") or 0),
        "52w_low": float(price_info.get("yearLow") or 0),
        "volume": int(trade_info.get("totalTradedVolume") or 0),
        "traded_value": float(trade_info.get("totalTradedValue") or 0),
        "market_cap": float(trade_info.get("totalMarketCap") or 0),
        "pe_ratio": sec_info.get("pdSymbolPe"),
        "sector_pe": sec_info.get("pdSectorPe"),
        "sector": sec_info.get("sector"),
        "industry": sec_info.get("industryInfo") or sec_info.get("basicIndustry"),
        "macro": sec_info.get("macro"),
        "delivery_pct": sec_info.get("deliveryTotradedQuantity"),
        "daily_volatility": price_info.get("cmDailyVolatility"),
        "annual_volatility": price_info.get("cmAnnualVolatility"),
        "price_band": price_info.get("priceBand"),
        "isin": meta.get("isinCode"),
        "index_membership": sec_info.get("indexList", [])[:6],
        "last_update": payload.get("lastUpdateTime"),
        "source": "NSE",
    }


def get_nse_equity_meta(symbol: str) -> dict:
    """Fetch listing metadata for a symbol (company name, active series, F&O flag)."""
    symbol = symbol.upper().strip()
    data = _get_json(
        f"{_QUOTE_API}?functionName=getMetaData&symbol={symbol}",
        referer=_equity_referer(symbol),
        timeout=10,
    )
    return {
        "symbol": data.get("symbol", symbol),
        "company_name": data.get("companyName", ""),
        "active_series": data.get("activeSeries", []),
        "is_fno": str(data.get("isFNOSec", "")).lower() == "true",
        "source": "NSE",
    }


def _normalise_option_symbol(symbol: str) -> str:
    """Map assorted user spellings onto NSE's option-chain symbol vocabulary."""
    cleaned = symbol.upper().strip().replace(" ", "")
    aliases = {
        "NIFTY50": "NIFTY",
        "NIFTYBANK": "BANKNIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "NIFTYFINANCIALSERVICES": "FINNIFTY",
        "NIFTYMIDCAPSELECT": "MIDCPNIFTY",
    }
    return aliases.get(cleaned, cleaned)


def get_option_expiries(symbol: str) -> list:
    """Return the available option expiry dates (e.g. '21-Jul-2026') for a symbol."""
    symbol = _normalise_option_symbol(symbol)
    data = _get_json(
        f"https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}",
        referer="https://www.nseindia.com/option-chain",
    )
    return list(data.get("expiryDates", []))


def get_nse_option_chain(symbol: str, expiry: Optional[str] = None, strikes_around_atm: int = 8) -> dict:
    """
    Fetch a live option chain from NSE for an index or F&O stock.

    This replaces the previous Yahoo Finance implementation, which could never
    work: Yahoo carries no options data for Indian symbols (ticker.options is
    an empty tuple for ^NSEI, RELIANCE.NS and every other .NS ticker), so the
    trading agent was always reasoning over an empty chain.

    Args:
        symbol: NIFTY, BANKNIFTY, FINNIFTY, or an F&O stock symbol.
        expiry: Expiry as 'DD-Mon-YYYY'. Defaults to the nearest expiry.
        strikes_around_atm: How many strikes to return either side of ATM.

    Returns dict with: symbol, expiry, underlying_value, calls, puts, pcr,
    max_pain, total_call_oi, total_put_oi, available_expiries, source
    """
    symbol = _normalise_option_symbol(symbol)

    expiries = get_option_expiries(symbol)
    if not expiries:
        return {"error": f"No option contracts listed for '{symbol}'", "symbol": symbol}

    chosen_expiry = expiry if expiry in expiries else expiries[0]

    chain_type = "Indices" if symbol in INDEX_OPTION_SYMBOLS else "Equity"
    data = _get_json(
        f"https://www.nseindia.com/api/option-chain-v3"
        f"?type={chain_type}&symbol={symbol}&expiry={chosen_expiry}",
        referer="https://www.nseindia.com/option-chain",
        timeout=12,
    )

    records = data.get("records", {})
    rows = records.get("data", [])
    underlying = float(records.get("underlyingValue") or 0)

    if not rows:
        return {
            "error": f"NSE returned an empty option chain for {symbol} {chosen_expiry}",
            "symbol": symbol,
            "expiry": chosen_expiry,
            "available_expiries": expiries[:6],
        }

    # Totals must come from the FULL chain, before any ATM trimming.
    total_call_oi = sum(int(r.get("CE", {}).get("openInterest") or 0) for r in rows)
    total_put_oi = sum(int(r.get("PE", {}).get("openInterest") or 0) for r in rows)
    pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi else 0.0
    max_pain = _calculate_max_pain(rows, underlying)

    # Keep only the strikes near ATM — a full chain is ~90 strikes of noise
    # that bloats the prompt and buries the levels that actually matter.
    all_strikes = sorted({float(r.get("strikePrice", 0)) for r in rows})
    atm_strike = min(all_strikes, key=lambda s: abs(s - underlying)) if all_strikes else underlying
    atm_index = all_strikes.index(atm_strike) if atm_strike in all_strikes else 0
    lo = max(0, atm_index - strikes_around_atm)
    hi = min(len(all_strikes), atm_index + strikes_around_atm + 1)
    wanted = set(all_strikes[lo:hi])

    calls, puts = [], []
    for row in rows:
        strike = float(row.get("strikePrice", 0))
        if strike not in wanted:
            continue
        if row.get("CE"):
            calls.append(_format_leg(row["CE"], strike))
        if row.get("PE"):
            puts.append(_format_leg(row["PE"], strike))

    calls.sort(key=lambda c: c["strike"])
    puts.sort(key=lambda p: p["strike"])

    return {
        "symbol": symbol,
        "expiry": chosen_expiry,
        "underlying_value": underlying,
        "atm_strike": atm_strike,
        "calls": calls,
        "puts": puts,
        "pcr": pcr,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "max_pain": max_pain,
        "available_expiries": expiries[:6],
        "source": "NSE",
    }


def _format_leg(leg: dict, strike: float) -> dict:
    """Normalise one CE/PE leg of the NSE option chain."""
    return {
        "strike": strike,
        "lastPrice": float(leg.get("lastPrice") or 0),
        "bid": float(leg.get("bidprice") or 0),
        "ask": float(leg.get("askPrice") or 0),
        "volume": int(leg.get("totalTradedVolume") or 0),
        "openInterest": int(leg.get("openInterest") or 0),
        "changeInOI": int(leg.get("changeinOpenInterest") or 0),
        "impliedVolatility": round(float(leg.get("impliedVolatility") or 0), 2),
    }


def _calculate_max_pain(rows: list, underlying: float) -> float:
    """
    Max pain = the strike where option writers lose the least in aggregate,
    i.e. the strike minimising total in-the-money payout across all OI.
    """
    try:
        strikes = sorted({float(r.get("strikePrice", 0)) for r in rows})
        if not strikes:
            return round(underlying, 2)

        call_oi = {float(r["strikePrice"]): int(r.get("CE", {}).get("openInterest") or 0) for r in rows}
        put_oi = {float(r["strikePrice"]): int(r.get("PE", {}).get("openInterest") or 0) for r in rows}

        min_pain, max_pain_strike = float("inf"), underlying
        for expiry_price in strikes:
            pain = 0.0
            for strike in strikes:
                if expiry_price > strike:  # calls at this strike expire ITM
                    pain += (expiry_price - strike) * call_oi.get(strike, 0)
                if expiry_price < strike:  # puts at this strike expire ITM
                    pain += (strike - expiry_price) * put_oi.get(strike, 0)
            if pain < min_pain:
                min_pain, max_pain_strike = pain, expiry_price

        return round(float(max_pain_strike), 2)
    except Exception:
        return round(underlying, 2)

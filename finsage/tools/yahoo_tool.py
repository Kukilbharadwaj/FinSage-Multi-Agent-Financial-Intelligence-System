# tools/yahoo_tool.py
# yfinance wrapper for stock data, company profiles, mutual funds, and historical OHLCV.
# This is the fallback when NSE scraper fails, and the primary source for historical data.

import yfinance as yf
from typing import Optional
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


# Maps common Indian stock/index names to Yahoo Finance symbols
SYMBOL_MAP = {
    # Large Cap Stocks
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
    # Additional Large & Mid Cap
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    "TITAN": "TITAN.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "HUL": "HINDUNILVR.NS",
    "POWERGRID": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "COALINDIA": "COALINDIA.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "TATAPOWER": "TATAPOWER.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "DMART": "DMART.NS",
    "IRCTC": "IRCTC.NS",
    "HAL": "HAL.NS",
    "BEL": "BEL.NS",
    "TRENT": "TRENT.NS",
    "JIOFIN": "JIOFIN.NS",
    "VEDL": "VEDL.NS",
    "TATACHEM": "TATACHEM.NS",
    "M&M": "M&M.NS",
    "MAHINDRA": "M&M.NS",
    "DRREDDY": "DRREDDY.NS",
    "CIPLA": "CIPLA.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "SBILIFE": "SBILIFE.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    # Indices
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


def get_company_profile(symbol: str) -> dict:
    """
    Fetch detailed company fundamentals from Yahoo Finance.

    Returns dict with: sector, industry, market_cap, pe_ratio, eps, dividend_yield,
    book_value, revenue, net_income, debt_to_equity, promoter_holding, description, etc.
    """
    try:
        yahoo_symbol = _resolve_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)
        info = ticker.info

        return {
            "symbol": symbol.upper(),
            "name": info.get("longName") or info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "market_cap_formatted": _format_indian_number(info.get("marketCap", 0)),
            "pe_ratio": round(float(info.get("trailingPE") or info.get("forwardPE") or 0), 2),
            "forward_pe": round(float(info.get("forwardPE") or 0), 2),
            "eps": round(float(info.get("trailingEps") or 0), 2),
            "dividend_yield": round(float(info.get("dividendYield") or 0) * 100, 2),
            "book_value": round(float(info.get("bookValue") or 0), 2),
            "price_to_book": round(float(info.get("priceToBook") or 0), 2),
            "revenue": info.get("totalRevenue", 0),
            "revenue_formatted": _format_indian_number(info.get("totalRevenue", 0)),
            "net_income": info.get("netIncomeToCommon", 0),
            "profit_margin": round(float(info.get("profitMargins") or 0) * 100, 2),
            "operating_margin": round(float(info.get("operatingMargins") or 0) * 100, 2),
            "roe": round(float(info.get("returnOnEquity") or 0) * 100, 2),
            "debt_to_equity": round(float(info.get("debtToEquity") or 0), 2),
            "current_ratio": round(float(info.get("currentRatio") or 0), 2),
            "free_cash_flow": info.get("freeCashflow", 0),
            "52w_high": round(float(info.get("fiftyTwoWeekHigh") or 0), 2),
            "52w_low": round(float(info.get("fiftyTwoWeekLow") or 0), 2),
            "avg_volume": info.get("averageVolume", 0),
            "beta": round(float(info.get("beta") or 0), 2),
            "description": (info.get("longBusinessSummary") or "N/A")[:500],
            "source": "Yahoo Finance",
        }

    except Exception as e:
        raise Exception(f"Yahoo Finance company profile error for '{symbol}': {str(e)}")


def get_intraday_data(symbol: str) -> dict:
    """
    Fetch intraday data (5-minute candles for today) from Yahoo Finance.
    Used for intraday trading analysis.

    Returns dict with: close, high, low, volume, dates (as lists).
    """
    try:
        yahoo_symbol = _resolve_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period="5d", interval="5m")
        market_status = get_indian_market_status()

        if df.empty:
            return {
                "market_status": market_status,
                "error": "No intraday candles returned from Yahoo Finance",
            }

        last_candle = df.index[-1]
        if getattr(last_candle, "tzinfo", None) is not None:
            last_candle_ist = last_candle.tz_convert("Asia/Kolkata")
        else:
            # Some providers return naive timestamps. Treat as IST to avoid false UTC conversion.
            last_candle_ist = last_candle.replace(tzinfo=ZoneInfo("Asia/Kolkata"))

        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        candle_delay_minutes = max(0, int((now_ist - last_candle_ist.to_pydatetime()).total_seconds() // 60))
        is_live_data = bool(market_status.get("is_open")) and candle_delay_minutes <= 20

        return {
            "close": df["Close"].tolist(),
            "high": df["High"].tolist(),
            "low": df["Low"].tolist(),
            "volume": df["Volume"].tolist(),
            "dates": [d.strftime("%Y-%m-%d %H:%M") for d in df.index],
            "interval": "5m",
            "last_price": round(float(df["Close"].iloc[-1]), 2),
            "day_high": round(float(df["High"].max()), 2),
            "day_low": round(float(df["Low"].min()), 2),
            "total_volume": int(df["Volume"].sum()),
            "vwap": round(float((df["Close"] * df["Volume"]).sum() / df["Volume"].sum()), 2) if df["Volume"].sum() > 0 else 0,
            "last_candle_time_ist": last_candle_ist.strftime("%Y-%m-%d %H:%M"),
            "candle_delay_minutes": candle_delay_minutes,
            "is_live_data": is_live_data,
            "market_status": market_status,
        }

    except Exception as e:
        raise Exception(f"Yahoo Finance intraday error for '{symbol}': {str(e)}")


def get_options_chain(symbol: str) -> dict:
    """
    Fetch options chain data from Yahoo Finance.
    Returns calls and puts for the nearest expiry.
    """
    try:
        yahoo_symbol = _resolve_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)

        # Get available expiry dates
        expiry_dates = ticker.options
        if not expiry_dates:
            return {"error": "No options data available for this symbol"}

        # Get nearest expiry
        nearest_expiry = expiry_dates[0]
        opt_chain = ticker.option_chain(nearest_expiry)

        # Current stock price for ATM detection
        info = ticker.info
        current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)

        # Process calls — top 10 by volume near ATM
        calls_df = opt_chain.calls
        calls_df = calls_df.sort_values("volume", ascending=False).head(15)
        calls = []
        for _, row in calls_df.iterrows():
            calls.append({
                "strike": float(row.get("strike", 0)),
                "lastPrice": float(row.get("lastPrice", 0)),
                "bid": float(row.get("bid", 0)),
                "ask": float(row.get("ask", 0)),
                "volume": int(row.get("volume", 0)) if row.get("volume") else 0,
                "openInterest": int(row.get("openInterest", 0)) if row.get("openInterest") else 0,
                "impliedVolatility": round(float(row.get("impliedVolatility", 0)) * 100, 2),
            })

        # Process puts — top 10 by volume near ATM
        puts_df = opt_chain.puts
        puts_df = puts_df.sort_values("volume", ascending=False).head(15)
        puts = []
        for _, row in puts_df.iterrows():
            puts.append({
                "strike": float(row.get("strike", 0)),
                "lastPrice": float(row.get("lastPrice", 0)),
                "bid": float(row.get("bid", 0)),
                "ask": float(row.get("ask", 0)),
                "volume": int(row.get("volume", 0)) if row.get("volume") else 0,
                "openInterest": int(row.get("openInterest", 0)) if row.get("openInterest") else 0,
                "impliedVolatility": round(float(row.get("impliedVolatility", 0)) * 100, 2),
            })

        # Calculate Put-Call Ratio from full chain
        total_call_oi = int(opt_chain.calls["openInterest"].sum()) if "openInterest" in opt_chain.calls else 0
        total_put_oi = int(opt_chain.puts["openInterest"].sum()) if "openInterest" in opt_chain.puts else 0
        pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0

        # Max Pain calculation (simplified)
        max_pain = _calculate_max_pain(opt_chain, current_price)

        return {
            "symbol": symbol.upper(),
            "expiry": nearest_expiry,
            "current_price": current_price,
            "calls": calls,
            "puts": puts,
            "pcr": pcr,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "max_pain": max_pain,
            "available_expiries": list(expiry_dates[:5]),
            "source": "Yahoo Finance",
        }

    except Exception as e:
        raise Exception(f"Yahoo Finance options error for '{symbol}': {str(e)}")


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


def get_indian_market_status(now_ist: Optional[datetime] = None) -> dict:
    """Return NSE market open/close status in IST (09:15 to 15:30, Mon-Fri)."""
    tz = ZoneInfo("Asia/Kolkata")
    now = now_ist or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    market_open_time = time(9, 15)
    market_close_time = time(15, 30)

    is_weekday = now.weekday() < 5
    is_open_time = market_open_time <= now.time() <= market_close_time
    is_open = is_weekday and is_open_time

    if is_open:
        status = "open"
        reason = "Indian market session is live."
    elif not is_weekday:
        status = "closed"
        reason = "Weekend: Indian market is closed."
    elif now.time() < market_open_time:
        status = "closed"
        reason = "Pre-market: session has not opened yet."
    else:
        status = "closed"
        reason = "Post-market: session has ended for the day."

    # Compute next open time (simple weekday logic, ignores exchange holidays).
    next_open_date = now.date()
    if is_weekday and now.time() < market_open_time:
        next_open_date = now.date()
    else:
        next_open_date = now.date() + timedelta(days=1)
        while next_open_date.weekday() >= 5:
            next_open_date += timedelta(days=1)

    next_open_dt = datetime.combine(next_open_date, market_open_time, tzinfo=tz)
    next_close_dt = datetime.combine(now.date(), market_close_time, tzinfo=tz)

    return {
        "is_open": is_open,
        "status": status,
        "reason": reason,
        "timezone": "Asia/Kolkata",
        "current_time_ist": now.strftime("%Y-%m-%d %H:%M"),
        "next_open_ist": next_open_dt.strftime("%Y-%m-%d %H:%M"),
        "today_close_ist": next_close_dt.strftime("%Y-%m-%d %H:%M"),
    }


def _format_indian_number(num) -> str:
    """Format a number in Indian style (Cr, L)."""
    try:
        num = float(num)
        if num >= 1e12:
            return f"₹{num/1e7/1e5:,.2f} Lakh Cr"
        elif num >= 1e7:
            return f"₹{num/1e7:,.2f} Cr"
        elif num >= 1e5:
            return f"₹{num/1e5:,.2f} L"
        elif num > 0:
            return f"₹{num:,.0f}"
        return "N/A"
    except (ValueError, TypeError):
        return "N/A"


def _calculate_max_pain(opt_chain, current_price: float) -> float:
    """Calculate max pain strike price (simplified)."""
    try:
        calls = opt_chain.calls
        puts = opt_chain.puts
        strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))

        if not strikes:
            return current_price

        min_pain = float("inf")
        max_pain_strike = current_price

        for strike in strikes:
            total_pain = 0
            # Pain for call writers
            for _, row in calls.iterrows():
                if strike > row["strike"]:
                    total_pain += (strike - row["strike"]) * (row.get("openInterest") or 0)
            # Pain for put writers
            for _, row in puts.iterrows():
                if strike < row["strike"]:
                    total_pain += (row["strike"] - strike) * (row.get("openInterest") or 0)

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike

        return round(float(max_pain_strike), 2)
    except Exception:
        return round(current_price, 2)

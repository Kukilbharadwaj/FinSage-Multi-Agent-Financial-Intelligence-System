# tools/mf_tool.py
# Indian Mutual Fund data fetcher using mftool library.
# Free, no API key required. Pulls data from AMFI (Association of Mutual Funds in India).

from mftool import Mftool

# Singleton instance
_mf = None

def _get_mf():
    global _mf
    if _mf is None:
        _mf = Mftool()
    return _mf

# Popular Indian Mutual Fund scheme codes (AMFI codes)
MF_SCHEME_MAP = {
    # Large Cap / Index Funds
    "NIFTY 50 INDEX FUND": "120505",
    "UTI NIFTY 50": "120505",
    "HDFC INDEX NIFTY 50": "112243",
    "SBI NIFTY INDEX": "119597",
    "ICICI PRU NIFTY 50": "120837",
    # ELSS (Tax Saving)
    "MIRAE ASSET TAX SAVER": "118834",
    "QUANT ELSS TAX SAVER": "120823",
    "AXIS LONG TERM EQUITY": "112304",
    "SBI LONG TERM EQUITY": "105757",
    "HDFC TAXSAVER": "100190",
    # Flexi Cap / Multi Cap
    "PARAG PARIKH FLEXI CAP": "122639",
    "PPFCF": "122639",
    "HDFC FLEXI CAP": "100027",
    "SBI FLEXI CAP": "119598",
    "KOTAK FLEXI CAP": "112090",
    # Mid Cap
    "HDFC MID CAP": "100086",
    "SBI MAGNUM MIDCAP": "105760",
    "KOTAK EMERGING EQUITY": "112091",
    # Small Cap
    "SBI SMALL CAP": "125497",
    "NIPPON SMALL CAP": "113177",
    "QUANT SMALL CAP": "120828",
    # Debt / Liquid
    "HDFC LIQUID": "100084",
    "SBI LIQUID": "105756",
    "ICICI LIQUID": "101180",
    # Hybrid / Balanced
    "HDFC BALANCED ADVANTAGE": "100025",
    "ICICI BAF": "110367",
    "SBI BALANCED ADVANTAGE": "119607",
}


def _resolve_mf_code(query: str) -> str:
    """Try to find a mutual fund scheme code from user query."""
    query_upper = query.upper().strip()

    # Direct match
    if query_upper in MF_SCHEME_MAP:
        return MF_SCHEME_MAP[query_upper]

    # Partial match
    for name, code in MF_SCHEME_MAP.items():
        if query_upper in name or name in query_upper:
            return code

    # If it looks like a numeric code already
    if query.strip().isdigit():
        return query.strip()

    return ""


def get_mf_nav(scheme_code: str) -> dict:
    """
    Fetch latest NAV and scheme details for a mutual fund.

    Args:
        scheme_code: AMFI scheme code (numeric string)

    Returns:
        Dict with scheme_name, nav, date, scheme_code
    """
    try:
        data = _get_mf().get_scheme_quote(scheme_code)
        if not data:
            return {"error": f"No data found for scheme code {scheme_code}"}

        return {
            "scheme_name": data.get("scheme_name", "N/A"),
            "scheme_code": scheme_code,
            "nav": float(data.get("last_price", 0)),
            "date": data.get("last_updated", "N/A"),
            "scheme_type": data.get("scheme_type", "N/A"),
            "scheme_category": data.get("scheme_category", "N/A"),
            "fund_house": data.get("fund_house", "N/A"),
        }
    except Exception as e:
        return {"error": f"MF data fetch failed: {str(e)[:150]}"}


def get_mf_history(scheme_code: str) -> dict:
    """
    Fetch historical NAV data for a mutual fund.

    Returns dict with nav_values (list of {date, nav}) for performance calculation.
    """
    try:
        data = _get_mf().get_scheme_historical_nav(scheme_code, as_Dataframe=False)
        if not data or "data" not in data:
            return {"error": "No historical data available"}

        nav_list = data["data"][:365]  # Last 1 year of data points
        return {
            "scheme_code": scheme_code,
            "scheme_name": data.get("scheme_name", "N/A"),
            "nav_history": [
                {"date": item["date"], "nav": float(item["nav"])}
                for item in nav_list
                if item.get("nav") and item["nav"] != "N/A"
            ][:60],  # Keep last ~60 data points for analysis
        }
    except Exception as e:
        return {"error": f"MF history fetch failed: {str(e)[:150]}"}


def search_mf_schemes(query: str) -> list:
    """
    Search for mutual fund schemes by name.

    Returns list of dicts with scheme_code and scheme_name.
    """
    try:
        # First try our map
        code = _resolve_mf_code(query)
        if code:
            nav_data = get_mf_nav(code)
            if "error" not in nav_data:
                return [{"scheme_code": code, "scheme_name": nav_data.get("scheme_name", query)}]

        # Search via mftool
        results = _get_mf().get_scheme_codes(as_Dataframe=False)
        if not results:
            return []

        matches = []
        query_words = query.upper().split()
        for code, name in results.items():
            name_upper = name.upper()
            if all(word in name_upper for word in query_words):
                matches.append({"scheme_code": str(code), "scheme_name": name})
            if len(matches) >= 5:
                break

        return matches
    except Exception as e:
        return [{"error": f"MF search failed: {str(e)[:100]}"}]


def get_mf_details(query: str) -> dict:
    """
    High-level function: search for a fund, get NAV + history, calculate returns.

    This is the main entry point for the mutual fund agent.
    """
    try:
        # Try to resolve scheme code
        code = _resolve_mf_code(query)

        if not code:
            # Search for it
            schemes = search_mf_schemes(query)
            if schemes and "error" not in schemes[0]:
                code = schemes[0]["scheme_code"]
            else:
                return {"error": f"Could not find mutual fund matching '{query}'"}

        # Get current NAV
        nav_data = get_mf_nav(code)
        if "error" in nav_data:
            return nav_data

        # Get history for return calculation
        history = get_mf_history(code)
        returns = {}
        if "error" not in history and history.get("nav_history"):
            nav_list = history["nav_history"]
            current_nav = nav_data["nav"]

            # Calculate approximate returns
            if len(nav_list) >= 5:
                # 1 month (approx 22 trading days)
                idx_1m = min(22, len(nav_list) - 1)
                nav_1m = nav_list[idx_1m]["nav"]
                returns["1_month"] = round(((current_nav - nav_1m) / nav_1m) * 100, 2) if nav_1m > 0 else 0

            if len(nav_list) >= 15:
                # 3 months
                idx_3m = min(60, len(nav_list) - 1)
                nav_3m = nav_list[idx_3m]["nav"]
                returns["3_month"] = round(((current_nav - nav_3m) / nav_3m) * 100, 2) if nav_3m > 0 else 0

            if len(nav_list) >= 30:
                # 6 months
                idx_6m = min(len(nav_list) - 1, 30)
                nav_6m = nav_list[idx_6m]["nav"]
                returns["6_month"] = round(((current_nav - nav_6m) / nav_6m) * 100, 2) if nav_6m > 0 else 0

        result = {
            **nav_data,
            "returns": returns,
        }

        return result

    except Exception as e:
        return {"error": f"MF details fetch failed: {str(e)[:150]}"}

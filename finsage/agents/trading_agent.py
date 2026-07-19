# agents/trading_agent.py
# Intraday and Options/F&O trading analysis agent.
# Fetches live intraday data, options chain, and provides trading signals.
# Model: GROQ_REASONING (openai/gpt-oss-120b) — step-by-step trading strategy
#
# Stage 3 agent — reads market_analysis from shared state.
# Writes: state["trading_analysis_output"]
# Uses: RAG Agent (on-demand) for trading rules

from groq import Groq
from config.settings import settings
from config.models import GROQ_REASONING
from mcp_bridge import call_mcp_tool, is_mcp_enabled
from tools.yahoo_tool import get_intraday_data, get_options_chain, get_indian_market_status
from agents.rag_agent import retrieve_for_agent

# Vocabulary that makes a question an options/intraday question. Shared with
# the supervisor: whichever agents it picks, a query containing any of these
# must route here, because this is the only agent that reports market timing.
TRADING_KEYWORDS = (
    "option", "call", "put", "f&o", "fno", "strike", "premium",
    "straddle", "strangle", "iron condor", "nifty option",
    "banknifty option", "pcr", "max pain", "oi", "open interest",
    "intraday", "scalp", "expiry", "futures", "swing trade",
)


def is_trading_query(text: str) -> bool:
    """True when the question is about options / intraday / F&O."""
    return any(kw in (text or "").lower() for kw in TRADING_KEYWORDS)


def build_market_notice(market_status: dict) -> str:
    """
    Build the market-timing line that every trading answer opens with.

    This is assembled in Python, not asked of the LLM. Whether the NSE is open
    is a fact about the clock, and a model that "usually" mentions it is the
    wrong tool — a trader reading a live entry trigger at 4pm on a Saturday
    needs to be told the session is closed every single time, not most times.

    The question still gets answered; the notice only frames when it applies.
    """
    phase = market_status.get("phase")
    now_ist = market_status.get("current_time_ist", "")
    next_open = market_status.get("next_open_ist", "")
    next_open_day = market_status.get("next_open_day", "")
    day_name = market_status.get("day_name", "")
    hours = market_status.get("session_hours_ist", "09:15 – 15:30 IST, Monday to Friday")

    if phase == "open":
        return (
            f"🟢 **Market is OPEN** — {now_ist} IST ({day_name}). "
            f"Session runs {hours}, so today's close is at "
            f"{market_status.get('today_close_ist', '15:30')}."
        )

    if phase == "weekend":
        return (
            f"🔴 **Market is CLOSED — it's {day_name}.** NSE and BSE don't trade on "
            f"Saturdays or Sundays. Regular session is {hours}. "
            f"Next open: {next_open_day} {next_open} IST. "
            f"Option chain and price data below are from the last trading session, "
            f"so treat everything as preparation for the next session, not a live call."
        )

    if phase == "pre_market":
        return (
            f"🟡 **Market hasn't opened yet** — {now_ist} IST ({day_name}). "
            f"Trading starts at 09:15 IST ({hours}). "
            f"Levels below are from the previous close — wait for the open to confirm them."
        )

    if phase == "post_market":
        return (
            f"🔴 **Market is CLOSED for today** — it's {now_ist} IST ({day_name}), "
            f"and the session ended at 15:30. Regular hours are {hours}. "
            f"Next open: {next_open_day} {next_open} IST. "
            f"The analysis below still stands, but treat it as a plan for the next "
            f"session rather than something to act on right now."
        )

    # Unknown phase — say so rather than implying the market is tradeable.
    return (
        f"⚠️ **Market status could not be confirmed.** NSE trades {hours}. "
        f"Check your broker terminal before acting on anything below."
    )


def run(state: dict) -> dict:
    """
    Analyze intraday and F&O trading opportunities using live data.

    Reads from shared state:
        - market_analysis: summary, sentiment, volatility (if available)

    Writes state["trading_analysis_output"] with structured output:
        - analysis: full LLM trading analysis text
        - market_status: market open/closed status dict
        - options_data: options chain dict (if applicable)
    """
    try:
        entities = state.get("entities", {})
        raw_query = state.get("raw_query", "").lower()

        # Determine symbol
        symbol = entities.get("stock") or entities.get("index") or "NIFTY 50"

        # Get trading rules from RAG Agent (on-demand)
        rag_context = retrieve_for_agent(state, "trading")

        # ── Read upstream: market_analysis from shared state ──
        market_info = state.get("market_analysis") or {}
        market_summary = market_info.get("summary", "")
        market_sentiment = market_info.get("sentiment", "neutral")
        market_volatility = market_info.get("volatility", "medium")

        # Fetch intraday data
        intraday = {}
        try:
            if is_mcp_enabled():
                intraday = call_mcp_tool("intraday_data", {"symbol": symbol})
            else:
                intraday = get_intraday_data(symbol)
        except Exception as e:
            intraday = {"error": str(e)[:100]}

        # Fetch options chain if query is about options/F&O
        options_data = {}
        is_options_query = is_trading_query(raw_query)

        if is_options_query:
            try:
                if is_mcp_enabled():
                    options_data = call_mcp_tool("options_chain", {"symbol": symbol})
                else:
                    options_data = get_options_chain(symbol)
            except Exception as e:
                options_data = {"error": str(e)[:100]}

        market_status = (intraday or {}).get("market_status") or get_indian_market_status()
        is_market_open = bool(market_status.get("is_open"))
        market_notice = build_market_notice(market_status)

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

        data_parts.append(f"""MARKET STATUS (IST) — this is ground truth, do not contradict it:
    - Status: {market_status.get('status', 'unknown').upper()}
    - Is Open: {is_market_open}
    - Phase: {market_status.get('phase', 'unknown')}
    - Today: {market_status.get('day_name', 'N/A')} ({'weekend' if market_status.get('is_weekend') else 'weekday'})
    - Session Hours: {market_status.get('session_hours_ist', '09:15 - 15:30 IST, Mon-Fri')}
    - Current Time: {market_status.get('current_time_ist', 'N/A')}
    - Next Open: {market_status.get('next_open_day', '')} {market_status.get('next_open_ist', 'N/A')}
    - Today Close: {market_status.get('today_close_ist', 'N/A')}
    - Note: {market_status.get('reason', 'N/A')}""")

        # Add upstream market context
        if market_summary:
            data_parts.append(f"""MARKET ANALYSIS CONTEXT (from Market Agent):
- Summary: {market_summary[:300]}
- Sentiment: {market_sentiment}
- Volatility: {market_volatility}""")

        if options_data and "error" not in options_data:
            # Strikes arrive from NSE sorted around ATM. Show them as a single
            # aligned ladder so the model can read call/put OI side by side —
            # that pairing is what identifies support and resistance walls.
            calls_by_strike = {c["strike"]: c for c in options_data.get("calls", [])}
            puts_by_strike = {p["strike"]: p for p in options_data.get("puts", [])}
            atm = options_data.get("atm_strike")

            ladder_rows = []
            for strike in sorted(set(calls_by_strike) | set(puts_by_strike)):
                call = calls_by_strike.get(strike, {})
                put = puts_by_strike.get(strike, {})
                marker = "  <- ATM" if strike == atm else ""
                ladder_rows.append(
                    f"  {strike:>9,.0f} | "
                    f"CE ₹{call.get('lastPrice', 0):>8,.2f} OI {call.get('openInterest', 0):>9,} "
                    f"(chg {call.get('changeInOI', 0):>+8,}) IV {call.get('impliedVolatility', 0):>5.1f}% | "
                    f"PE ₹{put.get('lastPrice', 0):>8,.2f} OI {put.get('openInterest', 0):>9,} "
                    f"(chg {put.get('changeInOI', 0):>+8,}) IV {put.get('impliedVolatility', 0):>5.1f}%"
                    f"{marker}"
                )

            # Highest-OI strikes are the levels traders actually watch.
            top_call_wall = max(calls_by_strike.values(), key=lambda c: c.get("openInterest", 0), default={})
            top_put_wall = max(puts_by_strike.values(), key=lambda p: p.get("openInterest", 0), default={})

            data_parts.append(f"""LIVE NSE OPTION CHAIN for {symbol} (Expiry: {options_data.get('expiry', 'N/A')}):
- Underlying (spot): ₹{options_data.get('underlying_value', 'N/A')}
- ATM Strike: ₹{atm}
- Put-Call Ratio (PCR): {options_data.get('pcr', 'N/A')}
- Max Pain: ₹{options_data.get('max_pain', 'N/A')}
- Total Call OI: {options_data.get('total_call_oi', 0):,} | Total Put OI: {options_data.get('total_put_oi', 0):,}
- Highest Call OI (resistance): ₹{top_call_wall.get('strike', 'N/A')} with {top_call_wall.get('openInterest', 0):,} OI
- Highest Put OI (support): ₹{top_put_wall.get('strike', 'N/A')} with {top_put_wall.get('openInterest', 0):,} OI

Strike ladder around ATM:
{chr(10).join(ladder_rows)}

Read PCR > 1 as put-heavy (supportive), PCR < 0.7 as call-heavy (resistive).
Use the OI walls as the real support/resistance, and max pain as the expiry magnet.""")

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

    If market is CLOSED (weekend, before 09:15, or after 15:30):
    - Say so plainly in the first line, including WHY (Saturday/Sunday, pre-market, or session ended).
    - Still answer the question fully — analyse the chain, the levels, the setup.
    - Frame it as "next session preparation" plus an "opening 15-min observation plan".
    - No immediate execution command, and note that the data is from the last session.

    If market is OPEN:
    - You can give actionable setup like "Buy only above X" / "Sell below Y".
    - Time window can be short (for example 5-15 min), but no guaranteed return claims.

Be specific with prices in ₹. This is for educational purposes only."""

        analysis_text = ""

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

            analysis_text = response.choices[0].message.content.strip()

        except Exception as llm_error:
            analysis_text = (
                f"Trading analysis could not be completed: {str(llm_error)[:100]}. "
                "Please use a professional trading terminal for real-time analysis."
            )

        # ── Write structured output to communication bus ──
        state["trading_analysis_output"] = {
            "analysis": analysis_text,
            "market_status": market_status,
            # Prepended verbatim to the final answer by the synthesis agent, so
            # open/closed never depends on the model remembering to say it.
            "market_notice": market_notice,
            "options_data": options_data if options_data and "error" not in options_data else None,
        }

        state["trace"].append(
            f"trading_agent → {symbol} "
            f"{'options + intraday' if is_options_query else 'intraday'} analysis ({market_status.get('status', 'unknown')})"
            + (f" (market: {market_sentiment})" if market_summary else "")
        )

    except Exception as e:
        # Market status is a clock lookup, not a network call — it should still
        # be reported even when the data fetch or the LLM leg failed.
        try:
            fallback_status = get_indian_market_status()
        except Exception:
            fallback_status = {}

        state["trading_analysis_output"] = {
            "analysis": f"Trading agent error: {str(e)[:200]}",
            "market_status": fallback_status,
            "market_notice": build_market_notice(fallback_status),
            "options_data": None,
        }
        state["trace"].append(f"trading_agent → ERROR: {str(e)[:100]}")

    return state

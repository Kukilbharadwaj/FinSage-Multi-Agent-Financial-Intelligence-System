# agents/synthesis_agent.py
# Combines all agent outputs into the final user-facing answer.
# Model: GROQ_STANDARD (llama-3.3-70b-versatile)
#
# Final stage — reads ALL structured analysis dicts + review_output.
# Writes: state["recommendation"], state["confidence"]
#
# Tone note: the previous version asked for a fixed heading template per intent
# and required the model to print "**Confidence:** N%" in the body, which is
# where the robotic feel came from — every answer looked like a filled-in form,
# and an internal metric leaked into the prose. Confidence is now computed from
# the review gate and returned as an API field instead of being written out.

from llm import Groq

from agents.memory import format_history
from config.models import GROQ_STANDARD
from config.settings import settings

# Shared voice for every answer, regardless of topic.
_BASE_VOICE = """You are FinSage — a sharp, friendly Indian financial advisor talking to one person.

How you write:
- Talk like a knowledgeable friend, not a report generator. Warm, direct, confident.
- Open with the actual answer to what they asked. No preamble, no restating the question.
- Use "you" and "your". Contractions are good. Vary your sentence length.
- Use headings ONLY when the answer genuinely has distinct parts. A short question deserves a short answer — two or three sentences is often perfect.
- Bold the numbers that matter. Always write rupees as ₹ with Indian formatting (₹1,25,000).
- Be specific. "Invest around ₹15,000/month in a Nifty 50 index fund" beats "consider index funds".
- Close with the single most useful next step, phrased naturally.

Hard rules:
- Use ONLY the data provided below. If a number is missing, say so plainly — never invent prices, NAVs or returns.
- Never promise or guarantee returns. Describe likelihoods and ranges instead.
- Do NOT print a confidence score, and do NOT mention agents, tools, pipelines or internal data sources.
- Do NOT append a disclaimer — one is added automatically afterwards.

Continuing a conversation:
- If earlier turns are shown, you are mid-conversation. Answer the new question as a reply, not as a fresh start.
- Reuse what they already told you — income, holdings, risk appetite — instead of asking again.
- Don't re-explain what you already covered. Reference it in a few words and move on to what's new."""

# Per-intent guidance on WHAT to cover — deliberately not a rigid template.
_INTENT_GUIDANCE = {
    "trading": """This is a trading question, so be practical and risk-first.

Cover, in whatever order reads best:
- Whether the market is open right now, and what that means for acting today.
- Your actual read on the setup, with the levels that matter (trigger, stop-loss, target).
- If the market is CLOSED: frame everything as a plan for the next session. No "buy now" calls.
- If an option chain is provided: use the real PCR, max pain and OI walls to justify your levels.
- Position sizing and max acceptable loss — this matters more than the entry.
- Work in SEBI's finding that roughly 9 out of 10 F&O traders lose money, naturally, not as a bolted-on warning.""",
    "mutual_fund": """This is a mutual fund question.

Cover what's relevant:
- A straight verdict on the fund or category, with the reasoning.
- What the returns actually tell you, and what they don't.
- A concrete SIP amount and time horizon when you have income data to base it on.
- Direct plan over regular, and why the expense ratio gap compounds.
- Who this suits and who it doesn't.""",
    "tax": """This is a tax question. Precision matters most here.

Cover:
- The direct answer, with the final rupee figure up front.
- The actual arithmetic, step by step, so they can verify it.
- Current rules: STCG on equity 20%, LTCG on equity 12.5% above the ₹1.25L annual exemption.
- Anything legal they can still do to reduce the liability.
- Note that a CA should confirm before filing.""",
    "salary": """This is a salary and budgeting question.

Cover:
- Where their money should go each month, in rupees, and make the numbers add up to their actual salary.
- Emergency fund and insurance before investments — this ordering is not optional.
- What to do in the first 30 days, concretely.
- One or two mistakes people at this income level usually make.""",
    "stock": """This is a stock or index question.

Cover:
- What the price is doing right now and what that means.
- Your genuine read — bullish, bearish or neutral — and the reasoning behind it.
- Valuation and fundamentals if you have them (P/E, ROE, debt).
- An entry zone or a wait condition, plus the level that would prove you wrong.
- What someone already holding should watch.
Give a real stance. Don't hedge into meaninglessness, but don't fake certainty either.""",
}
_INTENT_GUIDANCE["index"] = _INTENT_GUIDANCE["stock"]

_GENERAL_GUIDANCE = """Answer directly and practically.

Cover the answer itself, the key reasoning, what to do next, and anything genuinely
risky to watch out for. Keep it proportional — don't stretch a simple question into an essay."""


def _collect_context(state: dict) -> list:
    """Gather every populated agent output into labelled prompt sections."""
    parts = []

    salary = state.get("salary_analysis")
    if isinstance(salary, dict) and salary.get("plan"):
        parts.append(f"SALARY ANALYSIS:\n{salary['plan'][:900]}")

    tax = state.get("tax_analysis")
    if isinstance(tax, dict) and tax.get("tax_result"):
        parts.append(f"TAX CALCULATION:\n{tax['tax_result'][:900]}")

    market = state.get("market_analysis")
    if isinstance(market, dict):
        if market.get("summary"):
            parts.append(f"MARKET ANALYSIS:\n{market['summary'][:700]}")

        data = market.get("market_data") or {}
        if data and not data.get("error"):
            parts.append(
                f"LIVE PRICE DATA:\n"
                f"- {data.get('symbol', 'N/A')}: ₹{data.get('price', 'N/A')} "
                f"({data.get('change', 'N/A')}%)\n"
                f"- Day range: ₹{data.get('low', 'N/A')} – ₹{data.get('high', 'N/A')}\n"
                f"- 52-week range: ₹{data.get('52w_low', 'N/A')} – ₹{data.get('52w_high', 'N/A')}\n"
                f"- Source: {data.get('source', 'N/A')}"
            )

        profile = market.get("company_profile") or {}
        if profile.get("name"):
            parts.append(
                f"COMPANY FUNDAMENTALS:\n"
                f"- {profile.get('name')} | {profile.get('sector', 'N/A')}\n"
                f"- Market cap: {profile.get('market_cap_formatted', 'N/A')}\n"
                f"- P/E: {profile.get('pe_ratio', 'N/A')} | EPS: ₹{profile.get('eps', 'N/A')}\n"
                f"- ROE: {profile.get('roe', 'N/A')}% | Debt/Equity: {profile.get('debt_to_equity', 'N/A')}\n"
                f"- Profit margin: {profile.get('profit_margin', 'N/A')}%"
            )

    news = state.get("news_analysis")
    if isinstance(news, dict) and (news.get("sentiment_score") or news.get("key_events")):
        parts.append(
            f"NEWS SENTIMENT: {news.get('sentiment_score', 0)} ({news.get('market_mood', 'neutral')})\n"
            f"Key events: {news.get('key_events', 'none')}"
        )

    tech = state.get("technical_analysis")
    if isinstance(tech, dict):
        signals = tech.get("signals") or {}
        if signals:
            parts.append(
                f"TECHNICAL INDICATORS:\n"
                f"- Price ₹{signals.get('current_price', 'N/A')} | Trend: {signals.get('trend', 'N/A')}\n"
                f"- EMA20 ₹{signals.get('ema20', 'N/A')} | EMA50 ₹{signals.get('ema50', 'N/A')}\n"
                f"- RSI {signals.get('rsi', 'N/A')} | MACD {signals.get('macd', 'N/A')}\n"
                f"- Support ₹{signals.get('support', 'N/A')} | Resistance ₹{signals.get('resistance', 'N/A')}"
            )
        if tech.get("analysis"):
            parts.append(f"TECHNICAL READ:\n{tech['analysis'][:700]}")

    trading = state.get("trading_analysis_output")
    if isinstance(trading, dict):
        status = trading.get("market_status") or {}
        if status:
            parts.append(
                f"MARKET STATUS: {str(status.get('status', 'unknown')).upper()} "
                f"({status.get('reason', '')}) | Today: {status.get('day_name', 'N/A')} | "
                f"IST now: {status.get('current_time_ist', 'N/A')} | "
                f"Session hours: {status.get('session_hours_ist', '09:15-15:30 IST, Mon-Fri')} | "
                f"Next open: {status.get('next_open_day', '')} {status.get('next_open_ist', 'N/A')}\n"
                f"A market-status line is prepended to your answer automatically — do NOT "
                f"repeat it, but keep everything you say consistent with it."
            )

        options = trading.get("options_data") or {}
        if options and not options.get("error"):
            parts.append(
                f"LIVE OPTION CHAIN ({options.get('symbol')} {options.get('expiry')}):\n"
                f"- Spot ₹{options.get('underlying_value')} | ATM ₹{options.get('atm_strike')}\n"
                f"- PCR {options.get('pcr')} | Max pain ₹{options.get('max_pain')}\n"
                f"- Call OI {options.get('total_call_oi', 0):,} | Put OI {options.get('total_put_oi', 0):,}"
            )

        if trading.get("analysis"):
            parts.append(f"TRADING ANALYSIS:\n{trading['analysis'][:1100]}")

    mf = state.get("mf_analysis")
    if isinstance(mf, dict) and mf.get("analysis"):
        parts.append(f"MUTUAL FUND ANALYSIS:\n{mf['analysis'][:1100]}")

    general = state.get("general_finance_result")
    if isinstance(general, dict) and general.get("answer"):
        parts.append(
            f"FINANCIAL ANALYSIS ({general.get('topic', 'general')}):\n{general['answer'][:1100]}"
        )

    return parts


def run(state: dict) -> dict:
    """
    Final agent: combine every available output into one clear, human answer.

    Reads all *_analysis dicts plus review_output, writes recommendation
    and confidence.
    """
    try:
        intent = state.get("intent", "general")
        goal = state.get("goal", state.get("raw_query", ""))

        context_parts = _collect_context(state)
        context_string = "\n\n".join(context_parts) if context_parts else "No agent data available."

        review = state.get("review_output") or {}
        review_issues = review.get("issues", [])
        review_corrections = review.get("corrections", [])
        review_confidence = review.get("confidence_score", 70)
        review_approved = review.get("approved", True)

        # Surface data gaps as instructions so the model states them honestly
        # instead of papering over missing inputs with generic filler.
        caveats = ""
        if review_issues:
            caveats = "\n\nData gaps you must be upfront about (weave in naturally, don't list them mechanically):\n"
            caveats += "\n".join(f"- {issue}" for issue in review_issues[:5])
        if review_corrections:
            caveats += "\n\nHow to handle them:\n" + "\n".join(
                f"- {c}" for c in review_corrections[:5]
            )
        if not review_approved:
            caveats += "\n\nMost of the analysis failed. Be honest about what you couldn't determine and keep the answer short rather than padding it."

        guidance = _INTENT_GUIDANCE.get(intent, _GENERAL_GUIDANCE)
        system_prompt = f"{_BASE_VOICE}\n\n---\n\n{guidance}"

        transcript = format_history(state.get("conversation_history") or [], answer_chars=400)
        history_block = (
            f"Earlier in this conversation:\n{transcript}\n\n---\n\n" if transcript else ""
        )

        user_prompt = f"""{history_block}The person asked: "{state['raw_query']}"

What they're trying to figure out: {goal}

Here's everything the analysis gathered:

{context_string}
{caveats}

Write your answer to them now."""

        client = Groq(api_key=settings.GROQ_API_KEY)

        response = client.chat.completions.create(
            name="synthesis_llm",
            model=GROQ_STANDARD,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,   # a little warmth; 0.4 read flat and templated
            max_tokens=1600,
        )

        answer = response.choices[0].message.content.strip()

        # Whether the market is open is a clock fact, so it is prepended here
        # rather than left to the model. Someone asking about an option trade at
        # 3:30pm or on a Saturday still gets the full answer — they just get it
        # with the session status stated up front, every time.
        trading = state.get("trading_analysis_output")
        if isinstance(trading, dict):
            notice = trading.get("market_notice")
            if notice:
                answer = f"{notice}\n\n{answer}"

        state["recommendation"] = answer

        # Confidence now comes from the review gate's data-quality assessment
        # rather than asking the model to grade itself in the visible answer.
        confidence = int(review_confidence)
        if not review_approved:
            confidence = min(confidence, 40)
        state["confidence"] = max(0, min(100, confidence))

        state["trace"].append("synthesis → answer generated")

    except Exception as e:
        state["recommendation"] = (
            "I ran into a problem putting your answer together. Please try asking again — "
            "and if it keeps happening, a SEBI-registered advisor can help with the specifics."
        )
        state["confidence"] = 0
        state["trace"].append(f"synthesis → ERROR: {str(e)[:100]}")

    return state

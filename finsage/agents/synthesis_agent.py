# agents/synthesis_agent.py
# Combines all agent outputs into a final user answer.
# Model: GROQ_STANDARD (llama-3.3-70b-versatile)
#
# Stage 5 (final) — reads ALL structured analysis dicts + review_output.
# Writes: state["recommendation"], state["confidence"]

from groq import Groq
from config.settings import settings
from config.models import GROQ_STANDARD


def run(state: dict) -> dict:
    """
    Final agent: combines every available piece of information
    into one clear, formatted recommendation.

    Reads from shared state:
        - salary_analysis, tax_analysis, market_analysis, news_analysis,
          mf_analysis, trading_analysis_output, technical_analysis,
          general_finance_result (all structured dicts)
        - review_output: issues, corrections, confidence_score
        - execution_plan: the plan created by Supervisor
    """
    try:
        intent = state.get("intent", "general")
        goal = state.get("goal", state.get("raw_query", ""))

        # ── Collect all available outputs from communication bus ──
        context_parts = []

        # Salary analysis
        salary = state.get("salary_analysis")
        if salary and isinstance(salary, dict):
            plan = salary.get("plan", "")
            if plan:
                context_parts.append(f"**Salary Analysis:**\n{plan[:800]}")

        # Tax analysis
        tax = state.get("tax_analysis")
        if tax and isinstance(tax, dict):
            tax_result = tax.get("tax_result", "")
            if tax_result:
                context_parts.append(f"**Tax Calculation:**\n{tax_result[:800]}")

        # Market analysis
        market = state.get("market_analysis")
        if market and isinstance(market, dict):
            summary = market.get("summary", "")
            if summary:
                context_parts.append(f"**Market Analysis:**\n{summary[:600]}")

            # Company profile
            profile = market.get("company_profile", {})
            if profile and isinstance(profile, dict) and profile.get("name"):
                context_parts.append(
                    f"**Company Fundamentals:**\n"
                    f"- Name: {profile.get('name', 'N/A')} | Sector: {profile.get('sector', 'N/A')}\n"
                    f"- Market Cap: {profile.get('market_cap_formatted', 'N/A')}\n"
                    f"- P/E: {profile.get('pe_ratio', 'N/A')} | EPS: ₹{profile.get('eps', 'N/A')}\n"
                    f"- Dividend Yield: {profile.get('dividend_yield', 'N/A')}% | ROE: {profile.get('roe', 'N/A')}%\n"
                    f"- Debt/Equity: {profile.get('debt_to_equity', 'N/A')} | P/B: {profile.get('price_to_book', 'N/A')}\n"
                    f"- Profit Margin: {profile.get('profit_margin', 'N/A')}%"
                )

        # News analysis
        news = state.get("news_analysis")
        if news and isinstance(news, dict):
            score = news.get("sentiment_score", 0)
            mood = news.get("market_mood", "neutral")
            events = news.get("key_events", "")
            if score != 0 or events:
                context_parts.append(f"**News Sentiment:** {score} ({mood})\nKey Events: {events}")

        # Technical analysis
        tech = state.get("technical_analysis")
        if tech and isinstance(tech, dict):
            analysis = tech.get("analysis", "")
            signals = tech.get("signals", {})
            if analysis and "error" not in analysis.lower()[:20]:
                context_parts.append(f"**Technical Analysis:**\n{analysis[:800]}")
            elif signals:
                context_parts.append(
                    f"**Technical Indicators:** Trend={signals.get('trend','N/A')}, "
                    f"RSI={signals.get('rsi','N/A')}, EMA20={signals.get('ema20','N/A')}, "
                    f"Support={signals.get('support','N/A')}, Resistance={signals.get('resistance','N/A')}"
                )

        # Trading analysis
        trading = state.get("trading_analysis_output")
        if trading and isinstance(trading, dict):
            analysis = trading.get("analysis", "")
            if analysis:
                context_parts.append(f"**Trading Analysis:**\n{analysis[:1000]}")

        # Mutual fund analysis
        mf = state.get("mf_analysis")
        if mf and isinstance(mf, dict):
            analysis = mf.get("analysis", "")
            if analysis:
                context_parts.append(f"**Mutual Fund Analysis:**\n{analysis[:1000]}")

        # General finance result
        general = state.get("general_finance_result")
        if general and isinstance(general, dict):
            answer = general.get("answer", "")
            if answer:
                context_parts.append(f"**Financial Analysis ({general.get('topic', 'general')}):**\n{answer[:1000]}")

        # Data freshness
        freshness = state.get("data_freshness", "N/A")
        context_parts.append(f"**Data Freshness:** {freshness}")

        context_string = "\n\n".join(context_parts) if context_parts else "No agent data available."

        # ── Read review output ──
        review = state.get("review_output") or {}
        review_issues = review.get("issues", [])
        review_corrections = review.get("corrections", [])
        review_confidence = review.get("confidence_score", 70)
        review_approved = review.get("approved", True)

        review_note = ""
        if review_issues:
            review_note = "\n\nReview Agent Notes:\n" + "\n".join(f"- {issue}" for issue in review_issues)
        if review_corrections:
            review_note += "\n\nCorrections to incorporate:\n" + "\n".join(f"- {c}" for c in review_corrections)
        if not review_approved:
            review_note += "\n\n⚠️ Review Agent flagged critical issues — reduce confidence and add caveats."

        # ── Determine sentiment guidance ──
        news_data = state.get("news_analysis") or {}
        sentiment = news_data.get("sentiment_score")
        sentiment_note = ""
        if sentiment is not None:
            if sentiment > 0.2:
                sentiment_note = "News sentiment is positive — mention this as a supporting factor."
            elif sentiment < -0.2:
                sentiment_note = "News sentiment is negative — warn the user about adverse news."

        # ── Choose system prompt based on intent ──
        if intent == "trading":
            system_prompt = """You are a senior Indian trading expert giving a final, human-style analysis.
Do not sound robotic and do not force a rigid BUY/SELL template when market is closed.

Use this structure:

**Market Status:** OPEN or CLOSED, with one-line practical meaning

**What I Would Do Now:**
- conversational guidance based on live status

**Trade Plan:**
- Trigger level(s)
- Stop-loss
- Target zone
- Time window (only when market is OPEN)

**If-Then Scenarios:**
- If breakout holds...
- If reversal starts...

**Risk Guardrails:**
- max risk per trade
- no overtrading
- include SEBI point: 9 out of 10 F&O traders lose money

**Confidence:** number between 0 and 100 percent

**Disclaimer:** Educational only. Not SEBI-registered investment advice.

Critical rules:
1) If market is CLOSED, avoid "buy now" / "sell now".
2) If market is OPEN, give actionable trigger-based plan with realistic uncertainty.
3) No guaranteed returns or certainty claims."""

        elif intent == "mutual_fund":
            system_prompt = """You are a senior Indian mutual fund advisor writing in a human, practical style.
Avoid robotic labels like one-word recommendation outputs.

Use this structure:

**What This Means For You:**
- simple 2-3 line summary

**Fund View:**
- strengths
- concerns
- suitability (who it is good for)

**SIP Guidance:**
- suggested SIP range and time horizon
- when to increase, pause, or continue

**What To Do Next:**
- clear next 3 steps

**Confidence:** number between 0 and 100 percent

**Disclaimer:** Educational only. Not SEBI-registered investment advice."""

        elif intent == "tax":
            system_prompt = """You are a senior Indian tax mentor explaining clearly in plain language.
Do not give generic legal text only; make it user-specific and actionable.

Use this structure:

**Quick Answer:**
- direct answer to the tax question

**How The Tax Is Calculated:**
- short calculation steps with ₹ values where available

**What You Can Still Do:**
- legal optimization options (if any)

**Documents/Compliance Checklist:**
- what to keep ready for filing

**Confidence:** number between 0 and 100 percent

**Disclaimer:** Educational tax information only. Consult a qualified CA/tax professional for filing decisions."""

        elif intent == "salary":
            system_prompt = """You are a practical Indian personal finance coach.
Respond like a helpful human planner, not a template bot.

Use this structure:

**Your Money Snapshot:**
- short summary of current salary situation

**Monthly Plan (₹):**
- essentials
- savings/investments
- protection (insurance/emergency fund)
- lifestyle spending

**First 30 Days Action Plan:**
- specific steps in order

**Common Mistakes To Avoid:**
- 2-3 brief points

**Confidence:** number between 0 and 100 percent

**Disclaimer:** Educational financial guidance only."""

        elif intent in ("stock", "index"):
            system_prompt = """You are a senior Indian market analyst speaking in a human style.
Do not force one-word outputs like BUY/SELL/HOLD.
Give a nuanced stance based on data quality, market context, and risk.

Use this structure:

**What The Market Is Saying Right Now:**
- short plain-language summary

**My Read (Bullish/Bearish/Neutral with reason):**
- explain the stance with key data points

**Actionable Plan:**
- entry zone or wait condition
- invalidation/stop level
- upside/downside zone

**If You Already Hold:**
- what to monitor and when to exit/review

**Confidence:** number between 0 and 100 percent

**Disclaimer:** Educational only. Not SEBI-registered investment advice."""

        elif intent in ("insurance", "loan", "retirement", "gold", "crypto"):
            system_prompt = """You are a senior Indian financial advisor giving practical, human guidance.

Use this structure:

**What This Means For You:**
- concise summary

**Best Next Move:**
- practical recommendation with reasoning

**Plan You Can Follow:**
- step-by-step actions

**Watchouts:**
- key risks and mistakes

**Confidence:** number between 0 and 100 percent

**Disclaimer:** Educational guidance only. Consult a qualified professional before decisions."""

        else:
            system_prompt = """You are a senior Indian financial advisor.
Write in a natural, human tone with clear practical guidance.
Avoid robotic one-word recommendations.

Use this structure:

**Short Answer:**
- direct response in plain language

**Why:**
- 2-4 key reasons from available data

**What You Should Do Next:**
- specific next steps

**What To Watch:**
- risks/uncertainties

**Confidence:** number between 0 and 100 percent

**Disclaimer:** Educational only, not personalized professional advice."""

        # ── Build execution plan context ──
        plan = state.get("execution_plan", [])
        plan_text = ""
        if plan:
            plan_text = "\n\nExecution Plan followed:\n" + "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))

        user_prompt = f"""User's question: "{state['raw_query']}"
User's goal: {goal}
Intent detected: {intent}
{plan_text}

Here is all the data collected by our analysis agents:

{context_string}

{sentiment_note}
{review_note}

Generate the final recommendation using the intent-specific structure above. Keep it human, practical, and specific with numbers and ₹ amounts where available."""

        client = Groq(api_key=settings.GROQ_API_KEY)

        response = client.chat.completions.create(
            model=GROQ_STANDARD,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=2500,
        )

        recommendation = response.choices[0].message.content.strip()
        state["recommendation"] = recommendation

        # Extract confidence from the response
        try:
            for line in recommendation.split("\n"):
                if "**Confidence:**" in line:
                    conf_str = line.split("**Confidence:**")[1].strip()
                    conf_num = "".join(c for c in conf_str if c.isdigit())
                    if conf_num:
                        state["confidence"] = min(100, max(0, int(conf_num)))
                    break
            if state.get("confidence") is None:
                state["confidence"] = 50
        except Exception:
            state["confidence"] = 50

        # Adjust confidence based on Review Agent's assessment
        if review_confidence is not None:
            # Blend: 60% LLM confidence, 40% review confidence
            llm_conf = state.get("confidence", 50)
            blended = int(llm_conf * 0.6 + review_confidence * 0.4)
            state["confidence"] = max(0, min(100, blended))

        if not review_approved:
            state["confidence"] = min(state.get("confidence", 50), 40)

        state["trace"].append("synthesis_agent → recommendation generated")

    except Exception as e:
        state["recommendation"] = (
            f"Unable to generate recommendation: {str(e)[:200]}. "
            "Please try again or consult a SEBI-registered financial advisor."
        )
        state["confidence"] = 0
        state["trace"].append(f"synthesis_agent → ERROR: {str(e)[:100]}")

    return state

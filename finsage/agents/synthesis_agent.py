# agents/synthesis_agent.py
# Combines all agent outputs into a final user answer.
# Model: GROQ_STANDARD (llama-3.3-70b-versatile)

from groq import Groq
from config.settings import settings
from config.models import GROQ_STANDARD


def run(state: dict) -> dict:
    """
    Final agent: combines every available piece of information
    into one clear, formatted recommendation.
    """
    try:
        # Collect all available outputs
        context_parts = []

        # Market data summary
        market_data = state.get("market_data")
        if market_data and isinstance(market_data, dict):
            summary = market_data.get("summary", "")
            if summary:
                context_parts.append(f"**Market Data:**\n{summary}")

        # Technical signals
        tech = state.get("technical_signals")
        if tech and isinstance(tech, dict) and "error" not in tech:
            analysis = tech.get("analysis", "")
            if analysis:
                context_parts.append(f"**Technical Analysis:**\n{analysis[:800]}")
            else:
                context_parts.append(
                    f"**Technical Indicators:** Trend={tech.get('trend','N/A')}, "
                    f"RSI={tech.get('rsi','N/A')}, EMA20={tech.get('ema20','N/A')}, "
                    f"Support={tech.get('support','N/A')}, Resistance={tech.get('resistance','N/A')}"
                )

        # Sentiment
        sentiment = state.get("sentiment_score")
        if sentiment is not None:
            mood = "positive" if sentiment > 0.2 else ("negative" if sentiment < -0.2 else "neutral")
            context_parts.append(f"**News Sentiment:** {sentiment} ({mood})")

        # Tax result
        tax_result = state.get("tax_result")
        if tax_result:
            context_parts.append(f"**Tax Calculation:**\n{tax_result[:800]}")

        # Salary plan
        salary_plan = state.get("salary_plan")
        if salary_plan and isinstance(salary_plan, dict):
            plan = salary_plan.get("plan", "")
            if plan:
                context_parts.append(f"**Salary Plan:**\n{plan[:800]}")

        # Data freshness
        freshness = state.get("data_freshness", "N/A")
        context_parts.append(f"**Data Freshness:** {freshness}")

        context_string = "\n\n".join(context_parts) if context_parts else "No agent data available."

        # Determine sentiment guidance
        sentiment_note = ""
        if sentiment is not None:
            if sentiment > 0.2:
                sentiment_note = "News sentiment is positive — mention this as a supporting factor."
            elif sentiment < -0.2:
                sentiment_note = "News sentiment is negative — warn the user about adverse news."

        # Determine confidence adjustment
        tech_trend = ""
        if tech and isinstance(tech, dict):
            tech_trend = tech.get("trend", "")
        confidence_note = ""
        if tech_trend == "bearish" and sentiment is not None and sentiment < -0.2:
            confidence_note = (
                "Both technical trend and news sentiment are negative. "
                "Reduce confidence and add a strong risk warning."
            )

        system_prompt = """You are a senior Indian financial advisor giving a final recommendation.
Format your response using this EXACT structure:

**Summary:** one sentence describing the action

**Recommendation:** BUY / SELL / HOLD / INVEST / WAIT (pick one word)

**Confidence:** number between 0 and 100 percent

**Key Reasons:**
- reason one
- reason two
- reason three

**Action Plan:**
specific steps the user should take now

**Risks to Watch:**
- risk one
- risk two

**Disclaimer:** This is AI-generated financial information for educational purposes only. Not SEBI-registered investment advice. Consult a SEBI-registered financial advisor before making investment decisions. Past performance does not guarantee future returns."""

        user_prompt = f"""User's question: "{state['raw_query']}"
Intent detected: {state.get('intent', 'general')}

Here is all the data collected by our analysis agents:

{context_string}

{sentiment_note}
{confidence_note}

Generate the final recommendation using the exact format specified. Be specific with numbers and ₹ amounts."""

        client = Groq(api_key=settings.GROQ_API_KEY)

        response = client.chat.completions.create(
            model=GROQ_STANDARD,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=2000,
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

        state["trace"].append("synthesis_agent → recommendation generated")

    except Exception as e:
        state["recommendation"] = (
            f"Unable to generate recommendation: {str(e)[:200]}. "
            "Please try again or consult a SEBI-registered financial advisor."
        )
        state["confidence"] = 0
        state["trace"].append(f"synthesis_agent → ERROR: {str(e)[:100]}")

    return state

# agents/mutual_fund_agent.py
# Indian Mutual Fund analysis agent.
# Fetches fund details, NAV, returns, and provides investment analysis.
# Model: GROQ_REASONING (qwen/qwen3-32b) — detailed fund analysis
#
# Stage 3 agent — reads salary_analysis, tax_analysis, market_analysis.
# Writes: state["mf_analysis"]
# Uses: RAG Agent (on-demand) for MF rules

from groq import Groq
from config.settings import settings
from config.models import GROQ_REASONING
from mcp_bridge import call_mcp_tool, is_mcp_enabled
from agents.rag_agent import retrieve_for_agent


def run(state: dict) -> dict:
    """
    Analyze mutual fund queries using mftool data, RAG context, and upstream agent data.

    Reads from shared state:
        - salary_analysis: investable_income, risk_profile (if available)
        - tax_analysis: remaining_80c, tax_saving_opportunities (if available)
        - market_analysis: sentiment, timing_recommendation (if available)

    Writes state["mf_analysis"] with structured output:
        - mutual_fund_data: raw fund data dict
        - sip_recommendation: text recommendation
        - analysis: full LLM analysis text
    """
    try:
        entities = state.get("entities", {})
        fund_name = entities.get("fund_name") or ""

        # Get MF rules from RAG Agent (on-demand)
        rag_context = retrieve_for_agent(state, "mutual_fund")

        # ── Read upstream: salary, tax, market analysis from shared state ──
        salary_info = state.get("salary_analysis") or {}
        investment_capacity = salary_info.get("investable_income")
        risk_profile = salary_info.get("risk_profile", "moderate")
        monthly_savings = salary_info.get("monthly_savings")

        tax_info = state.get("tax_analysis") or {}
        remaining_80c = tax_info.get("remaining_80c")
        tax_opportunities = tax_info.get("tax_saving_opportunities", [])

        market_info = state.get("market_analysis") or {}
        market_sentiment = market_info.get("sentiment", "neutral")
        market_timing = market_info.get("timing_recommendation", "")
        market_volatility = market_info.get("volatility", "medium")

        # Build upstream context string for the LLM
        upstream_context = ""
        if investment_capacity:
            upstream_context += f"""
From Salary Analysis:
- Investable Income: ₹{investment_capacity:,.0f}/month
- Monthly Savings: ₹{monthly_savings:,.0f}/month
- Risk Profile: {risk_profile}
"""
        if remaining_80c:
            upstream_context += f"""
From Tax Analysis:
- Remaining 80C Room: ₹{remaining_80c:,.0f}
- Tax Saving Options: {', '.join(tax_opportunities)}
"""
        if market_sentiment != "neutral" or market_timing:
            upstream_context += f"""
From Market Analysis:
- Market Sentiment: {market_sentiment}
- Volatility: {market_volatility}
- Timing View: {market_timing}
"""

        # Try to fetch fund data from mftool
        mf_data = {}
        try:
            from tools.mf_tool import get_mf_details

            if fund_name:
                if is_mcp_enabled():
                    mf_data = call_mcp_tool("mf_details", {"query": fund_name})
                else:
                    mf_data = get_mf_details(fund_name)
            else:
                # Try to extract fund name from raw query
                query = state.get("raw_query", "")
                if is_mcp_enabled():
                    mf_data = call_mcp_tool("mf_details", {"query": query})
                else:
                    mf_data = get_mf_details(query)
        except Exception as e:
            mf_data = {"error": str(e)[:150]}

        # Build analysis prompt
        system_message = """You are an expert Indian mutual fund advisor. Analyze fund data and provide 
clear investment guidance. Always mention direct vs regular plan, expense ratio importance, 
and SIP vs lump sum. Use ₹ symbol for amounts. Be specific and actionable."""

        # Build data section
        mf_info = ""
        if mf_data and "error" not in mf_data:
            returns = mf_data.get("returns", {})
            mf_info = f"""
FUND DATA:
- Name: {mf_data.get('scheme_name', 'N/A')}
- Fund House: {mf_data.get('fund_house', 'N/A')}
- Category: {mf_data.get('scheme_category', 'N/A')}
- Type: {mf_data.get('scheme_type', 'N/A')}
- Current NAV: ₹{mf_data.get('nav', 'N/A')}
- NAV Date: {mf_data.get('date', 'N/A')}
- 1-Month Return: {returns.get('1_month', 'N/A')}%
- 3-Month Return: {returns.get('3_month', 'N/A')}%
- 6-Month Return: {returns.get('6_month', 'N/A')}%
"""
        else:
            mf_info = f"\nNote: Could not fetch specific fund data. Error: {mf_data.get('error', 'Unknown')}\n"

        user_message = f"""User's question: "{state['raw_query']}"

{mf_info}

{upstream_context if upstream_context else "No upstream analysis available — provide general MF guidance."}

MUTUAL FUND RULES (from knowledge base):
{rag_context[:800]}

Provide a comprehensive mutual fund analysis:

1. **Fund Overview**: Category, fund house reputation, investment style
2. **Performance Assessment**: Recent returns vs category average, consistency
3. **Key Metrics**: Expense ratio importance, exit load, lock-in period (if any)
4. **SIP Recommendation**: Suggested SIP amount based on {f'investable income of ₹{investment_capacity:,.0f}' if investment_capacity else 'fund type'}
5. **Direct vs Regular**: Always recommend direct plan and explain why
6. **Tax-Saving Angle**: {f'Remaining 80C room is ₹{remaining_80c:,.0f} — recommend ELSS if applicable' if remaining_80c else 'Mention 80C benefits if ELSS fund'}
7. **Market Timing**: {f'Current market is {market_sentiment} with {market_volatility} volatility — factor into SIP vs lump sum advice' if market_sentiment != 'neutral' else 'Standard SIP advice'}
8. **Risk Assessment**: Fund-specific risks and mitigation
9. **Alternative Options**: 2-3 similar funds to consider for comparison

If the user asks a general MF question (not about a specific fund), provide educational guidance 
on fund selection, SIP strategy, or the specific topic they asked about.

Be specific with numbers. This is for educational purposes only."""

        sip_recommendation = ""
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

            # Extract SIP recommendation if mentioned
            for line in analysis_text.split("\n"):
                if "sip" in line.lower() and "₹" in line:
                    sip_recommendation = line.strip()
                    break
            if not sip_recommendation and investment_capacity:
                sip_recommendation = f"Suggested SIP: ₹{min(investment_capacity * 0.5, 25000):,.0f}/month"

        except Exception as llm_error:
            analysis_text = (
                f"MF analysis could not be completed: {str(llm_error)[:100]}. "
                "Please check fund details on AMFIIndia.com or Moneycontrol."
            )

        # ── Write structured output to communication bus ──
        state["mf_analysis"] = {
            "mutual_fund_data": mf_data,
            "sip_recommendation": sip_recommendation,
            "analysis": analysis_text,
        }

        upstream_str = []
        if investment_capacity:
            upstream_str.append("salary")
        if remaining_80c:
            upstream_str.append("tax")
        if market_sentiment != "neutral":
            upstream_str.append("market")

        state["trace"].append(
            f"mutual_fund_agent → analyzed '{fund_name or 'general MF query'}'"
            + (f" (with {'+'.join(upstream_str)} context)" if upstream_str else "")
        )

    except Exception as e:
        state["mf_analysis"] = {
            "mutual_fund_data": {"error": str(e)[:200]},
            "sip_recommendation": "",
            "analysis": f"Mutual fund agent error: {str(e)[:200]}",
        }
        state["trace"].append(f"mutual_fund_agent → ERROR: {str(e)[:100]}")

    return state

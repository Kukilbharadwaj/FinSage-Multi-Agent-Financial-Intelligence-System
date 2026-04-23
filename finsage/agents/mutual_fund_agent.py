# agents/mutual_fund_agent.py
# Indian Mutual Fund analysis agent.
# Fetches fund details, NAV, returns, and provides investment analysis.
# Model: GROQ_REASONING (qwen/qwen3-32b) — detailed fund analysis

from groq import Groq
from config.settings import settings
from config.models import GROQ_REASONING
from rag.knowledge_base import query_kb


def run(state: dict) -> dict:
    """
    Analyze mutual fund queries using mftool data and RAG context.

    Fetches fund NAV, returns, and uses knowledge base for MF rules.
    Produces detailed fund analysis and recommendation.
    """
    try:
        entities = state.get("entities", {})
        fund_name = entities.get("fund_name") or ""

        # Get MF rules from RAG
        rag_context = query_kb(
            "mutual fund SIP NAV direct plan regular plan expense ratio "
            "ELSS index fund flexi cap mid cap small cap"
        )
        state["rag_context"] = rag_context

        # Try to fetch fund data from mftool
        mf_data = {}
        try:
            from tools.mf_tool import get_mf_details, search_mf_schemes

            if fund_name:
                mf_data = get_mf_details(fund_name)
            else:
                # Try to extract fund name from raw query
                query = state.get("raw_query", "")
                mf_data = get_mf_details(query)

            if "error" not in mf_data:
                state["mutual_fund_data"] = mf_data
            else:
                state["mutual_fund_data"] = mf_data
        except Exception as e:
            state["mutual_fund_data"] = {"error": str(e)[:150]}

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

MUTUAL FUND RULES (from knowledge base):
{rag_context[:800]}

Provide a comprehensive mutual fund analysis:

1. **Fund Overview**: Category, fund house reputation, investment style
2. **Performance Assessment**: Recent returns vs category average, consistency
3. **Key Metrics**: Expense ratio importance, exit load, lock-in period (if any)
4. **SIP Recommendation**: Suggested SIP amount based on fund type
5. **Direct vs Regular**: Always recommend direct plan and explain why
6. **Portfolio Fit**: Where this fund fits in a diversified portfolio
7. **Risk Assessment**: Fund-specific risks and mitigation
8. **Alternative Options**: 2-3 similar funds to consider for comparison

If the user asks a general MF question (not about a specific fund), provide educational guidance 
on fund selection, SIP strategy, or the specific topic they asked about.

Be specific with numbers. This is for educational purposes only."""

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

            # Store the analysis as the recommendation directly for MF queries
            state["mutual_fund_data"] = state.get("mutual_fund_data") or {}
            if isinstance(state["mutual_fund_data"], dict):
                state["mutual_fund_data"]["analysis"] = response.choices[0].message.content.strip()

        except Exception as llm_error:
            if isinstance(state.get("mutual_fund_data"), dict):
                state["mutual_fund_data"]["analysis"] = (
                    f"MF analysis could not be completed: {str(llm_error)[:100]}. "
                    "Please check fund details on AMFIIndia.com or Moneycontrol."
                )

        state["trace"].append(f"mutual_fund_agent → analyzed '{fund_name or 'general MF query'}'")

    except Exception as e:
        state["mutual_fund_data"] = {"error": str(e)[:200]}
        state["trace"].append(f"mutual_fund_agent → ERROR: {str(e)[:100]}")

    return state

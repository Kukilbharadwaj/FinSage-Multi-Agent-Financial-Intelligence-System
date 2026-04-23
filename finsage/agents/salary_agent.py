# agents/salary_agent.py
# Monthly budget plan for salary management queries.
# Model: GROQ_STANDARD (llama-3.3-70b-versatile)

from groq import Groq
from config.settings import settings
from config.models import GROQ_STANDARD
from rag.knowledge_base import query_kb


def run(state: dict) -> dict:
    """
    Generate a detailed monthly budget and investment plan based on salary.
    Uses RAG context for Indian financial planning rules.
    """
    try:
        entities = state.get("entities", {})

        # Extract salary amount — default to 20000 if not found
        salary = entities.get("amount")
        if salary is not None:
            try:
                salary = float(salary)
            except (ValueError, TypeError):
                salary = 20000.0
        else:
            salary = 20000.0

        # Get financial planning rules from knowledge base
        rag_context = query_kb("salary budget emergency fund SIP India PPF ELSS NPS")

        system_prompt = (
            "You are an expert Indian personal finance advisor. You know PPF (7.1%), "
            "ELSS (3-year lock-in, 80C), NPS (extra 50K under 80CCD1B), SIP in index funds, "
            "term insurance, health insurance, FD, RD, and emergency funds. "
            "Create practical monthly budget plans with exact rupee amounts. "
            "All amounts must add up exactly to the salary."
        )

        user_prompt = f"""The user's monthly salary is Rs {salary:,.0f}.

Relevant Indian financial planning rules:
---
{rag_context}
---

Create a complete monthly budget and investment plan with these sections:

1. **Fixed Needs Estimate**: Rent, groceries, utilities, transport
2. **Emergency Fund**: Target amount (3-6 months expenses) and monthly contribution
3. **Term Insurance**: Coverage amount and approximate monthly premium
4. **Health Insurance**: Personal health cover note
5. **SIP Allocation**: Monthly SIP with recommended fund type
6. **Tax Saving under 80C**: Monthly toward ELSS/PPF/NPS
7. **Remaining Discretionary Budget**: Amount left for wants

IMPORTANT: All amounts must add up exactly to Rs {salary:,.0f}. Use Rs symbol."""

        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            response = client.chat.completions.create(
                model=GROQ_STANDARD,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=1500,
            )

            plan_text = response.choices[0].message.content.strip()
            state["salary_plan"] = {"plan": plan_text}

        except Exception as llm_error:
            state["salary_plan"] = {
                "plan": (
                    f"Could not generate salary plan: {str(llm_error)[:100]}. "
                    f"General recommendation for Rs {salary:,.0f}: "
                    "Follow the 50-30-20 rule."
                )
            }

        state["trace"].append(f"salary_agent -> plan for Rs {salary:,.0f}")

    except Exception as e:
        state["salary_plan"] = {"plan": f"Salary planning error: {str(e)[:200]}"}
        state["trace"].append(f"salary_agent -> ERROR: {str(e)[:100]}")

    return state

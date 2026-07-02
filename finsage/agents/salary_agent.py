# agents/salary_agent.py
# Monthly budget plan for salary management queries.
# Model: GROQ_STANDARD (llama-3.3-70b-versatile)
#
# Stage 1 agent — no upstream dependencies.
# Writes: state["salary_analysis"] (read by tax_agent, mutual_fund_agent)
# Uses: RAG Agent (on-demand) for financial planning rules

import json
from groq import Groq
from config.settings import settings
from config.models import GROQ_STANDARD
from agents.rag_agent import retrieve_for_agent


def run(state: dict) -> dict:
    """
    Generate a detailed monthly budget and investment plan based on salary.

    Writes state["salary_analysis"] with structured output for downstream agents:
        - annual_salary, monthly_salary: raw numbers
        - monthly_savings, investable_income: estimated from plan
        - risk_profile: inferred from salary level
        - plan: full text from LLM
    """
    try:
        entities = state.get("entities", {})

        # Extract salary amount — default to 20000 if not found
        salary = entities.get("amount")
        if salary is not None:
            try:
                salary = float(salary)
                # If amount looks like annual (>= 100000), convert to monthly
                if salary >= 100000:
                    annual = salary
                    salary = salary / 12
                else:
                    annual = salary * 12
            except (ValueError, TypeError):
                salary = 20000.0
                annual = salary * 12
        else:
            salary = 20000.0
            annual = salary * 12

        # Get financial planning rules from RAG Agent (on-demand)
        rag_context = retrieve_for_agent(state, "salary")

        system_prompt = (
            "You are an expert Indian personal finance advisor. You know PPF (7.1%), "
            "ELSS (3-year lock-in, 80C), NPS (extra 50K under 80CCD1B), SIP in index funds, "
            "term insurance, health insurance, FD, RD, and emergency funds. "
            "Create practical monthly budget plans with exact rupee amounts. "
            "All amounts must add up exactly to the salary."
        )

        user_prompt = f"""The user's monthly salary is Rs {salary:,.0f} (annual: Rs {annual:,.0f}).

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

IMPORTANT: All amounts must add up exactly to Rs {salary:,.0f}. Use Rs symbol.

At the end, provide a JSON block with these estimates:
```json
{{"monthly_savings": <number>, "investable_income": <number>, "risk_profile": "conservative|moderate|aggressive"}}
```"""

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

            # Try to extract structured data from the LLM response
            monthly_savings = salary * 0.3  # default 30%
            investable_income = salary * 0.2  # default 20%
            risk_profile = "moderate"

            try:
                if "```json" in plan_text:
                    json_block = plan_text.split("```json")[1].split("```")[0].strip()
                    parsed = json.loads(json_block)
                    monthly_savings = float(parsed.get("monthly_savings", monthly_savings))
                    investable_income = float(parsed.get("investable_income", investable_income))
                    risk_profile = parsed.get("risk_profile", risk_profile)
            except (json.JSONDecodeError, IndexError, ValueError):
                pass  # Use defaults

            # Write structured output to communication bus
            state["salary_analysis"] = {
                "annual_salary": annual,
                "monthly_salary": salary,
                "monthly_savings": monthly_savings,
                "investable_income": investable_income,
                "risk_profile": risk_profile,
                "plan": plan_text,
            }

        except Exception as llm_error:
            state["salary_analysis"] = {
                "annual_salary": annual,
                "monthly_salary": salary,
                "monthly_savings": salary * 0.3,
                "investable_income": salary * 0.2,
                "risk_profile": "moderate",
                "plan": (
                    f"Could not generate salary plan: {str(llm_error)[:100]}. "
                    f"General recommendation for Rs {salary:,.0f}: "
                    "Follow the 50-30-20 rule."
                ),
            }

        state["trace"].append(f"salary_agent → plan for Rs {salary:,.0f}/month (Rs {annual:,.0f}/year)")

    except Exception as e:
        state["salary_analysis"] = {
            "annual_salary": 0,
            "monthly_salary": 0,
            "monthly_savings": 0,
            "investable_income": 0,
            "risk_profile": "moderate",
            "plan": f"Salary planning error: {str(e)[:200]}",
        }
        state["trace"].append(f"salary_agent → ERROR: {str(e)[:100]}")

    return state

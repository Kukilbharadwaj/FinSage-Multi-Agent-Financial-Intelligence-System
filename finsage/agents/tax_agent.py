# agents/tax_agent.py
# Indian tax rules: STCG, LTCG, 80C calculations using RAG + qwen3 reasoning.
# Model: GROQ_REASONING (openai/gpt-oss-120b) — step-by-step tax math
#
# Stage 2 agent — reads salary_analysis from shared state.
# Writes: state["tax_analysis"] (read by mutual_fund_agent)
# Uses: RAG Agent (on-demand) for tax rules

from llm import Groq
from config.settings import settings
from config.models import GROQ_REASONING
from agents.rag_agent import retrieve_for_agent


def run(state: dict) -> dict:
    """
    Calculate Indian tax implications using RAG context and qwq reasoning.

    Reads from shared state:
        - salary_analysis: annual_salary, monthly_savings, investable_income (if available)

    Writes state["tax_analysis"] with structured output:
        - tax_result: full calculation text
        - remaining_80c: estimated remaining 80C deduction room
        - tax_saving_opportunities: list of instruments
        - effective_rate: estimated effective tax rate string
    """
    try:
        # Get relevant tax rules from RAG Agent (on-demand)
        rag_context = retrieve_for_agent(state, "tax")

        entities = state.get("entities", {})
        amount = entities.get("amount")

        # ── Read upstream: salary_analysis from shared state ──
        salary_info = state.get("salary_analysis") or {}
        annual_salary = salary_info.get("annual_salary")
        monthly_savings = salary_info.get("monthly_savings")
        investable_income = salary_info.get("investable_income")

        salary_context = ""
        if annual_salary:
            salary_context = f"""
Salary information (from salary analysis):
- Annual Salary: ₹{annual_salary:,.0f}
- Monthly Savings: ₹{monthly_savings:,.0f}
- Investable Income: ₹{investable_income:,.0f}
"""

        # Build messages for the reasoning model
        system_message = "You are an Indian tax calculation expert. Calculate taxes step by step with precise rupee amounts."

        user_message = f"""
The user asked: "{state['raw_query']}"

Here are the relevant Indian tax rules and regulations:
---
{rag_context}
---

{f'The amount mentioned is ₹{amount}.' if amount else 'No specific amount was mentioned.'}
{salary_context}

Please calculate the tax step by step:

Step 1: Identify if the gains mentioned are Short Term Capital Gains (STCG — holding period under 12 months for equity) or Long Term Capital Gains (LTCG — holding period 12 months or more for equity). If the user mentions a specific holding period, use that. If not mentioned, analyze both scenarios.

Step 2: Apply the correct tax rate:
- STCG on equity (listed shares/equity MF): 20% flat rate (post July 2024 budget)
- LTCG on equity: 12.5% on gains above ₹1,25,000 annual exemption (post July 2024 budget)
- Debt fund gains: Added to income and taxed at slab rate

Step 3: Check if any Section 80C deductions could apply to reduce overall tax burden. List applicable instruments. Estimate remaining 80C room (₹1.5 lakh limit).

Step 4: Calculate the final tax amount in rupees. Show the math clearly with ₹ symbol.

Step 5: Suggest one actionable tax optimization strategy. For example:
- If holding period is close to 12 months, suggest waiting to convert STCG to LTCG
- If near end of financial year, suggest tax-loss harvesting
- If gains are below ₹1.25 lakh, mention LTCG exemption

Step 6: Remind the user that this is an estimate and they should consult a Chartered Accountant (CA) for official tax filing.

Be specific with rupee calculations. Use ₹ symbol. Show all math steps clearly.

At the end, provide a JSON block:
```json
{{"remaining_80c": <number or null>, "tax_saving_opportunities": ["ELSS", "PPF", "NPS"], "effective_rate": "<percentage string>"}}
```"""

        # Defaults for structured output
        remaining_80c = 150000  # full 1.5L if unknown
        tax_saving_opps = ["ELSS", "PPF", "NPS", "Tax-saving FD"]
        effective_rate = "N/A"
        tax_result_text = ""

        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            response = client.chat.completions.create(
                name="tax_llm",
                model=GROQ_REASONING,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.6,
                max_tokens=2000,
                reasoning_format="hidden",
            )

            tax_result_text = response.choices[0].message.content.strip()

            # Try to extract structured data from response
            try:
                if "```json" in tax_result_text:
                    json_block = tax_result_text.split("```json")[1].split("```")[0].strip()
                    import json
                    parsed = json.loads(json_block)
                    if parsed.get("remaining_80c") is not None:
                        remaining_80c = float(parsed["remaining_80c"])
                    if parsed.get("tax_saving_opportunities"):
                        tax_saving_opps = parsed["tax_saving_opportunities"]
                    if parsed.get("effective_rate"):
                        effective_rate = parsed["effective_rate"]
            except Exception:
                pass  # Use defaults

        except Exception as llm_error:
            tax_result_text = (
                f"Tax calculation could not be completed due to an error: {str(llm_error)[:100]}. "
                "Please consult a Chartered Accountant for accurate tax computation."
            )

        # Write structured output to communication bus
        state["tax_analysis"] = {
            "tax_result": tax_result_text,
            "remaining_80c": remaining_80c,
            "tax_saving_opportunities": tax_saving_opps,
            "effective_rate": effective_rate,
        }

        state["trace"].append(
            "tax_agent → calculated via reasoning model + RAG"
            + (" + salary context" if annual_salary else "")
        )

    except Exception as e:
        state["tax_analysis"] = {
            "tax_result": f"Tax agent error: {str(e)[:200]}. Please consult a CA.",
            "remaining_80c": 150000,
            "tax_saving_opportunities": ["ELSS", "PPF", "NPS"],
            "effective_rate": "N/A",
        }
        state["trace"].append(f"tax_agent → ERROR: {str(e)[:100]}")

    return state

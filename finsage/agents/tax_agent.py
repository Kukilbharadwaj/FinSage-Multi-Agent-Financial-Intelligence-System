# agents/tax_agent.py
# Indian tax rules: STCG, LTCG, 80C calculations using RAG + qwen3 reasoning.
# Model: GROQ_REASONING (qwen/qwen3-32b) — step-by-step tax math

from groq import Groq
from config.settings import settings
from config.models import GROQ_REASONING
from rag.knowledge_base import query_kb


def run(state: dict) -> dict:
    """
    Calculate Indian tax implications using RAG context and qwq reasoning.

    Retrieves relevant tax rules from FAISS, then uses qwq-32b for
    step-by-step tax calculation.
    """
    try:
        # Get relevant tax rules from knowledge base
        rag_context = query_kb(state["raw_query"])
        state["rag_context"] = rag_context

        entities = state.get("entities", {})
        amount = entities.get("amount")

        # Build messages for qwen3-32b reasoning
        system_message = "You are an Indian tax calculation expert. Calculate taxes step by step with precise rupee amounts."

        user_message = f"""

The user asked: "{state['raw_query']}"

Here are the relevant Indian tax rules and regulations:
---
{rag_context}
---

{f'The amount mentioned is ₹{amount}.' if amount else 'No specific amount was mentioned.'}

Please calculate the tax step by step:

Step 1: Identify if the gains mentioned are Short Term Capital Gains (STCG — holding period under 12 months for equity) or Long Term Capital Gains (LTCG — holding period 12 months or more for equity). If the user mentions a specific holding period, use that. If not mentioned, analyze both scenarios.

Step 2: Apply the correct tax rate:
- STCG on equity (listed shares/equity MF): 20% flat rate (post July 2024 budget)
- LTCG on equity: 12.5% on gains above ₹1,25,000 annual exemption (post July 2024 budget)
- Debt fund gains: Added to income and taxed at slab rate

Step 3: Check if any Section 80C deductions could apply to reduce overall tax burden. List applicable instruments.

Step 4: Calculate the final tax amount in rupees. Show the math clearly with ₹ symbol.

Step 5: Suggest one actionable tax optimization strategy. For example:
- If holding period is close to 12 months, suggest waiting to convert STCG to LTCG
- If near end of financial year, suggest tax-loss harvesting
- If gains are below ₹1.25 lakh, mention LTCG exemption

Step 6: Remind the user that this is an estimate and they should consult a Chartered Accountant (CA) for official tax filing.

Be specific with rupee calculations. Use ₹ symbol. Show all math steps clearly."""

        # Call qwen3-32b reasoning model
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            response = client.chat.completions.create(
                model=GROQ_REASONING,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.6,
                max_tokens=2000,
                reasoning_format="hidden",
            )

            state["tax_result"] = response.choices[0].message.content.strip()

        except Exception as llm_error:
            state["tax_result"] = (
                f"Tax calculation could not be completed due to an error: {str(llm_error)[:100]}. "
                "Please consult a Chartered Accountant for accurate tax computation."
            )

        state["trace"].append("tax_agent → calculated via qwen3 + RAG")

    except Exception as e:
        state["tax_result"] = f"Tax agent error: {str(e)[:200]}. Please consult a CA."
        state["rag_context"] = ""
        state["trace"].append(f"tax_agent → ERROR: {str(e)[:100]}")

    return state

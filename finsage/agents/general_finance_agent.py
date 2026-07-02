# agents/general_finance_agent.py
# Handles insurance, loans, retirement, gold, crypto, and general financial literacy.
# Uses RAG context for Indian-specific rules.
# Model: GROQ_STANDARD (llama-3.3-70b-versatile)
#
# Stage 1 agent — no upstream dependencies.
# Writes: state["general_finance_result"] as structured dict
# Uses: RAG Agent (on-demand) for domain-specific rules

from groq import Groq
from config.settings import settings
from config.models import GROQ_STANDARD
from agents.rag_agent import retrieve_for_agent


def run(state: dict) -> dict:
    """
    Handle insurance, loan, retirement, gold, crypto, and general finance queries.
    Uses RAG context for Indian-specific rules and guidelines.

    Writes state["general_finance_result"] with structured output:
        - answer: full LLM response text
        - topic: the specific sub-topic (insurance/loan/retirement/gold/crypto/general)
    """
    try:
        intent = state.get("intent", "general")
        entities = state.get("entities", {})
        amount = entities.get("amount")

        # Get relevant context from RAG Agent (on-demand)
        rag_context = retrieve_for_agent(state, "general_finance")

        # Build intent-specific system prompt
        intent_prompts = {
            "insurance": "You are an Indian insurance expert. Advise on term insurance, health insurance, and 80D benefits. Always recommend term over ULIP/endowment.",
            "loan": "You are an Indian loan and EMI expert. Calculate EMIs, explain tax benefits (Section 24, 80C, 80E), and advise on loan management.",
            "retirement": "You are an Indian retirement planning expert. Advise on NPS, pension, corpus calculation, and senior citizen schemes.",
            "gold": "You are an Indian gold investment expert. Always recommend SGB over physical gold. Explain tax implications clearly.",
            "crypto": "You are an Indian crypto taxation expert. Explain the 30% flat tax, 1% TDS, and no loss offset rules clearly.",
            "general": "You are a comprehensive Indian financial advisor. Provide clear, actionable guidance on any financial topic.",
        }

        system_prompt = intent_prompts.get(intent, intent_prompts["general"])

        user_prompt = f"""User's question: "{state['raw_query']}"

{f'Amount mentioned: ₹{amount}' if amount else 'No specific amount mentioned.'}

Relevant Indian financial rules and regulations:
---
{rag_context[:1000]}
---

Provide a detailed, step-by-step answer:

1. **Direct Answer**: Address the user's specific question clearly
2. **Detailed Explanation**: Break down the relevant rules, calculations, or concepts
3. **Tax Implications**: Any applicable tax benefits or liabilities
4. **Actionable Steps**: What the user should do right now
5. **Important Warnings**: Any risks or common mistakes to avoid
6. **Disclaimer**: Remind to consult a qualified professional

Use ₹ symbol. Be specific with numbers. Show calculations where applicable."""

        try:
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

            answer_text = response.choices[0].message.content.strip()

            state["general_finance_result"] = {
                "answer": answer_text,
                "topic": intent,
            }

        except Exception as llm_error:
            state["general_finance_result"] = {
                "answer": (
                    f"Could not generate analysis: {str(llm_error)[:100]}. "
                    "Please consult a qualified financial advisor for personalized guidance."
                ),
                "topic": intent,
            }

        state["trace"].append(f"general_finance_agent → {intent} query answered")

    except Exception as e:
        state["general_finance_result"] = {
            "answer": f"General finance agent error: {str(e)[:200]}",
            "topic": "general",
        }
        state["trace"].append(f"general_finance_agent → ERROR: {str(e)[:100]}")

    return state

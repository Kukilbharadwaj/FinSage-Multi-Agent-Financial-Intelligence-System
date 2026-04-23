# agents/intent_agent.py
# Detects what the user is asking and extracts entities.
# Model: GROQ_FAST (llama-3.1-8b-instant) — fast classification, under 200ms

import json
from groq import Groq
from config.settings import settings
from config.models import GROQ_FAST


def run(state: dict) -> dict:
    """
    Classify user intent and extract entities from the query.

    Intent: one of salary, stock, index, tax, general
    Entities: {stock, amount, index} — may be empty/null

    Always appends to state["trace"]. Never crashes.
    """
    try:
        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = """You are a financial query classifier for Indian users. 
Classify the user's question into exactly ONE of these intents:
- "salary" — if the user mentions salary, monthly income, budget planning, savings from salary
- "stock" — if the user asks about a specific company stock (Reliance, TCS, Infosys, etc.)
- "index" — if the user asks about Nifty, Nifty 50, Sensex, BankNifty, or market index
- "tax" — if the user asks about capital gains tax, STCG, LTCG, 80C, tax saving, tax on profits
- "general" — if the query is about general market conditions, news, or doesn't fit above categories

Also extract these entities from the query:
- "stock": the stock symbol if mentioned (use NSE symbol like RELIANCE, TCS, INFY). Set null if not mentioned.
- "amount": any rupee amount mentioned as a number (e.g., 20000 for ₹20,000). Set null if not mentioned.
- "index": the index name if mentioned (NIFTY 50, SENSEX, BANKNIFTY). Set null if not mentioned.

Respond ONLY with valid JSON in this exact format, nothing else:
{"intent": "...", "entities": {"stock": null, "amount": null, "index": null}}"""

        response = client.chat.completions.create(
            model=GROQ_FAST,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["raw_query"]},
            ],
            temperature=0.0,
            max_tokens=200,
        )

        raw_response = response.choices[0].message.content.strip()

        # Try to parse JSON from the response
        # Handle cases where model wraps JSON in markdown code blocks
        json_str = raw_response
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        result = json.loads(json_str)

        state["intent"] = result.get("intent", "general")
        state["entities"] = result.get("entities", {})

        # Validate intent is one of allowed values
        valid_intents = {"salary", "stock", "index", "tax", "general"}
        if state["intent"] not in valid_intents:
            state["intent"] = "general"

    except (json.JSONDecodeError, Exception) as e:
        # If anything fails, default to general intent
        state["intent"] = "general"
        state["entities"] = {}
        state["trace"].append(f"intent_agent → ERROR: {str(e)[:100]}, defaulting to general")
        return state

    state["trace"].append(f"intent_agent → {state['intent']}")
    return state

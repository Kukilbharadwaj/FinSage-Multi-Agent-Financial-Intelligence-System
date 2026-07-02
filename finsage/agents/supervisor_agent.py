# agents/supervisor_agent.py
# Supervisor Agent — replaces the old Intent Agent.
#
# Receives the user query and uses LLM (GROQ_FAST) to:
#   1. Understand the user's goal
#   2. Determine which agents are required (multi-select)
#   3. Create a numbered execution plan
#   4. Extract entities (stock, amount, index, fund_name)
#
# The Supervisor does NOT execute agents — it only decides and plans.
# The graph's stage dispatcher handles execution order.

import json
from groq import Groq
from config.settings import settings
from config.models import GROQ_FAST


# All valid agent names the Supervisor can select
VALID_AGENTS = {
    "salary", "tax", "market", "news", "trading",
    "mutual_fund", "technical", "general_finance",
}


def run(state: dict) -> dict:
    """
    Supervisor: classify user intent, select agents, and create execution plan.

    Writes to state:
        - goal: human-readable description of what the user wants
        - intent: primary intent string (backward compat)
        - entities: {stock, amount, index, fund_name}
        - selected_agents: list of agent names to execute
        - execution_plan: numbered list of plan steps
    """
    try:
        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = """You are the Supervisor of a multi-agent Indian financial assistant.

Your job: analyze the user's question and decide which specialist agents should handle it.

Available agents:
- "salary" — salary breakdown, monthly budget, savings plan, investment allocation
- "tax" — capital gains tax (STCG/LTCG), income tax, 80C/80D deductions, ITR
- "market" — live stock/index prices, company fundamentals, market data
- "news" — financial news headlines, market sentiment
- "trading" — intraday trading, options/F&O, swing trading, option chains
- "mutual_fund" — mutual fund NAV, SIP, fund analysis, ELSS
- "technical" — technical indicators (EMA, RSI, MACD), chart analysis
- "general_finance" — insurance, loans, retirement/NPS, gold, crypto, general literacy

Rules for agent selection:
1. Select ONLY the agents needed for this specific query (minimum 1, maximum 5).
2. If query mentions a specific stock or index → include "market" + "technical".
3. If query mentions trading, options, F&O → include "trading" + "market".
4. If query mentions both salary AND tax → include both (they share data).
5. If query involves salary AND investing → include "salary" + "mutual_fund".
6. If query involves tax-saving investments → include "tax" + "mutual_fund".
7. Include "news" when current market context would help the answer.
8. For general topics (insurance, loans, retirement, gold, crypto) → use "general_finance".
9. Do NOT select agents that are irrelevant to the query.

You must also extract entities:
- "stock": NSE symbol if a specific company is mentioned (RELIANCE, TCS, INFY, etc.). null if none.
- "amount": any rupee amount as a number (e.g., 50000 for ₹50,000 or 1200000 for 12 LPA). null if none.
- "index": index name if mentioned (NIFTY 50, SENSEX, BANKNIFTY). null if none.
- "fund_name": mutual fund name if mentioned. null if none.

Respond ONLY with valid JSON in this exact format, nothing else:
{
  "goal": "one sentence describing what the user wants",
  "intent": "primary intent (salary/stock/index/tax/trading/mutual_fund/insurance/loan/retirement/gold/crypto/general)",
  "selected_agents": ["agent1", "agent2"],
  "entities": {"stock": null, "amount": null, "index": null, "fund_name": null},
  "execution_plan": [
    "Step 1 description",
    "Step 2 description"
  ]
}"""

        response = client.chat.completions.create(
            model=GROQ_FAST,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["raw_query"]},
            ],
            temperature=0.0,
            max_tokens=500,
        )

        raw_response = response.choices[0].message.content.strip()

        # Parse JSON — handle markdown code block wrapping
        json_str = raw_response
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        result = json.loads(json_str)

        # Populate state from Supervisor output
        state["goal"] = result.get("goal", state["raw_query"])
        state["intent"] = result.get("intent", "general")
        state["entities"] = result.get("entities", {})
        state["execution_plan"] = result.get("execution_plan", [])

        # Validate and filter selected agents
        selected = result.get("selected_agents", [])
        state["selected_agents"] = [
            a for a in selected if a in VALID_AGENTS
        ]

        # Safety: if no valid agents selected, default based on intent
        if not state["selected_agents"]:
            intent = state["intent"]
            if intent in ("stock", "index"):
                state["selected_agents"] = ["market", "news", "technical"]
            elif intent == "tax":
                state["selected_agents"] = ["tax"]
            elif intent == "salary":
                state["selected_agents"] = ["salary"]
            elif intent == "trading":
                state["selected_agents"] = ["trading", "market"]
            elif intent == "mutual_fund":
                state["selected_agents"] = ["mutual_fund"]
            elif intent in ("insurance", "loan", "retirement", "gold", "crypto"):
                state["selected_agents"] = ["general_finance"]
            else:
                state["selected_agents"] = ["news"]

        # Ensure execution plan is not empty
        if not state["execution_plan"]:
            state["execution_plan"] = [
                f"Analyze query using {', '.join(state['selected_agents'])}",
                "Review outputs for consistency",
                "Generate recommendation",
            ]

    except (json.JSONDecodeError, Exception) as e:
        # Fallback: if Supervisor fails, default to general/news
        state["goal"] = state.get("raw_query", "")
        state["intent"] = "general"
        state["entities"] = {}
        state["selected_agents"] = ["news"]
        state["execution_plan"] = [
            "Fetch latest market news",
            "Generate general response",
        ]
        state["trace"].append(f"supervisor_agent → ERROR: {str(e)[:100]}, defaulting to news")
        return state

    agents_str = ", ".join(state["selected_agents"])
    state["trace"].append(f"supervisor_agent → goal: {state['goal'][:60]} | agents: [{agents_str}]")
    return state

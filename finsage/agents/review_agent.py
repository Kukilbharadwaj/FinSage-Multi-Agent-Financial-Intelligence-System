# agents/review_agent.py
# Review / Critic Agent — validates all agent outputs before synthesis.
#
# Runs as Stage 4 (after all domain agents). Uses GROQ_FAST for speed.
#
# Checks:
#   1. Plan completion: every execution_plan step has a corresponding output
#   2. Contradictions: e.g. salary says "invest aggressively" but market is bearish
#   3. Missing data: a selected agent's *_analysis is None or contains error
#   4. Logical consistency: tax amounts vs salary amounts
#   5. Unsupported claims: guaranteed returns, certainty language
#
# Output:
#   state["review_output"] = {
#       "issues": [...],
#       "corrections": [...],
#       "confidence_score": 0-100,
#       "approved": True/False
#   }

import json
from groq import Groq
from config.settings import settings
from config.models import GROQ_FAST


# Map agent names to the state keys they write
_AGENT_OUTPUT_KEYS = {
    "salary": "salary_analysis",
    "tax": "tax_analysis",
    "market": "market_analysis",
    "news": "news_analysis",
    "trading": "trading_analysis_output",
    "mutual_fund": "mf_analysis",
    "technical": "technical_analysis",
    "general_finance": "general_finance_result",
}


def _collect_outputs_summary(state: dict) -> str:
    """Build a text summary of all available agent outputs for the LLM."""
    parts = []
    selected = state.get("selected_agents", [])

    for agent_name in selected:
        key = _AGENT_OUTPUT_KEYS.get(agent_name)
        if not key:
            continue

        output = state.get(key)
        if output is None:
            parts.append(f"[{agent_name}]: NO OUTPUT (agent did not produce results)")
        elif isinstance(output, dict) and "error" in str(output):
            parts.append(f"[{agent_name}]: ERROR — {str(output)[:200]}")
        elif isinstance(output, dict):
            # Summarize the dict keys and truncated values
            summary_items = []
            for k, v in output.items():
                v_str = str(v)[:150] if v is not None else "None"
                summary_items.append(f"  {k}: {v_str}")
            parts.append(f"[{agent_name}]:\n" + "\n".join(summary_items))
        else:
            parts.append(f"[{agent_name}]: {str(output)[:300]}")

    return "\n\n".join(parts) if parts else "No agent outputs available."


def run(state: dict) -> dict:
    """
    Review Agent: validate all agent outputs before synthesis.

    Reads every *_analysis dict from state, checks for issues,
    and writes state["review_output"] with findings.
    """
    try:
        selected = state.get("selected_agents", [])
        plan = state.get("execution_plan", [])
        outputs_summary = _collect_outputs_summary(state)

        # Quick pre-check: identify obviously missing outputs
        pre_issues = []
        for agent_name in selected:
            key = _AGENT_OUTPUT_KEYS.get(agent_name)
            if key and state.get(key) is None:
                pre_issues.append(
                    f"Agent '{agent_name}' was selected but produced no output ({key} is empty)"
                )

        client = Groq(api_key=settings.GROQ_API_KEY)

        system_prompt = """You are a financial analysis reviewer. Your job is to check the quality and consistency of multiple agent outputs before they are combined into a final recommendation.

Check for:
1. CONTRADICTIONS: One agent says invest aggressively while another warns of bearish markets
2. MISSING DATA: An agent was supposed to run but produced no output or an error
3. LOGICAL CONSISTENCY: Tax amounts should be reasonable given salary; percentages should make sense
4. UNSUPPORTED CLAIMS: Any guaranteed returns or certainty language is a red flag
5. PLAN COMPLETION: Each step in the execution plan should have a corresponding output

Respond ONLY with valid JSON:
{
  "issues": ["list of problems found (empty if none)"],
  "corrections": ["suggested fixes for synthesis to incorporate"],
  "confidence_score": 75,
  "approved": true
}

confidence_score rules:
- 90-100: All agents succeeded, data is consistent, no contradictions
- 70-89: Minor issues (one agent had limited data, small inconsistencies)
- 50-69: Significant issues (missing agent output, notable contradictions)
- 0-49: Critical failures (most agents failed, major contradictions)

Set approved=false ONLY if there are critical contradictions that make the response unreliable."""

        user_prompt = f"""User's original question: "{state['raw_query']}"

Execution plan:
{chr(10).join(f'  {i+1}. {step}' for i, step in enumerate(plan))}

Selected agents: {', '.join(selected)}

Agent outputs:
{outputs_summary}

Pre-check issues found automatically:
{chr(10).join(f'  - {issue}' for issue in pre_issues) if pre_issues else '  None'}

Review these outputs and provide your assessment."""

        response = client.chat.completions.create(
            model=GROQ_FAST,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )

        raw_response = response.choices[0].message.content.strip()

        # Parse JSON response
        json_str = raw_response
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        result = json.loads(json_str)

        state["review_output"] = {
            "issues": result.get("issues", []) + pre_issues,
            "corrections": result.get("corrections", []),
            "confidence_score": max(0, min(100, int(result.get("confidence_score", 70)))),
            "approved": result.get("approved", True),
        }

    except (json.JSONDecodeError, Exception) as e:
        # If review fails, approve with lower confidence
        state["review_output"] = {
            "issues": [f"Review agent encountered an error: {str(e)[:100]}"],
            "corrections": [],
            "confidence_score": 50,
            "approved": True,
        }
        state["trace"].append(f"review_agent → ERROR: {str(e)[:100]}, approving with low confidence")
        return state

    issues_count = len(state["review_output"]["issues"])
    conf = state["review_output"]["confidence_score"]
    state["trace"].append(
        f"review_agent → {issues_count} issues, confidence {conf}%, "
        f"{'approved' if state['review_output']['approved'] else 'NOT approved'}"
    )
    return state

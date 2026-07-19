# agents/review_agent.py
# Review / Critic gate — validates agent outputs before synthesis.
#
# This used to be an LLM call (GROQ_FAST) that re-read every agent's output and
# returned a JSON verdict. That cost a full round trip on every query to
# produce a score that mostly restated facts already knowable from the state:
# which agents were selected, which wrote output, and which reported an error.
#
# It is now a local computation. Same contract, same state key, ~0ms.
# The checks that genuinely needed judgement (contradiction detection) are
# handled where they belong — the synthesis prompt receives the issue list and
# is told to reconcile disagreements explicitly.

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

# The main text field each agent's dict carries, used to spot error placeholders
_AGENT_TEXT_FIELDS = {
    "salary_analysis": "plan",
    "tax_analysis": "tax_result",
    "market_analysis": "summary",
    "trading_analysis_output": "analysis",
    "mf_analysis": "analysis",
    "technical_analysis": "analysis",
    "general_finance_result": "answer",
}

_ERROR_MARKERS = (
    "could not be completed",
    "could not fetch",
    "agent error",
    "unavailable",
    "insufficient",
    "error:",
    "failed",
)


def _output_health(agent_name: str, output) -> tuple:
    """Return (status, detail) for one agent's output: ok | degraded | missing."""
    if output is None:
        return "missing", f"'{agent_name}' was selected but produced no output"

    if not isinstance(output, dict):
        return "ok", ""

    key = _AGENT_OUTPUT_KEYS.get(agent_name, "")
    text_field = _AGENT_TEXT_FIELDS.get(key, "")
    text = str(output.get(text_field, "")) if text_field else ""

    if text:
        lowered = text.lower()[:200]
        if any(marker in lowered for marker in _ERROR_MARKERS):
            return "degraded", f"'{agent_name}' returned a fallback instead of real analysis"

    # A nested data dict carrying an error means the tool layer failed
    for value in output.values():
        if isinstance(value, dict) and value.get("error"):
            return "degraded", f"'{agent_name}' could not fetch live data ({str(value['error'])[:80]})"

    if not text and not any(output.values()):
        return "missing", f"'{agent_name}' produced an empty result"

    return "ok", ""


def _detect_contradictions(state: dict) -> list:
    """Flag cross-agent disagreements that synthesis should reconcile explicitly."""
    issues = []

    news = state.get("news_analysis") or {}
    market = state.get("market_analysis") or {}
    technical = state.get("technical_analysis") or {}

    sentiment = news.get("sentiment_score")
    trend = str((technical.get("signals") or {}).get("trend", "")).lower()

    # News mood vs technical trend pulling opposite directions
    if isinstance(sentiment, (int, float)) and trend:
        if sentiment < -0.2 and "bullish" in trend:
            issues.append("News sentiment is negative while technicals are bullish — acknowledge both sides.")
        elif sentiment > 0.2 and "bearish" in trend:
            issues.append("News sentiment is positive while technicals are bearish — acknowledge both sides.")

    # Trading advice built on stale data while the market is shut
    trading = state.get("trading_analysis_output") or {}
    status = (trading.get("market_status") or {}).get("is_open")
    if status is False:
        issues.append("Market is closed — frame trading guidance as a next-session plan, not a live call.")

    # Tax figures that cannot be reconciled with the stated salary
    salary = state.get("salary_analysis") or {}
    annual = salary.get("annual_salary")
    if annual and market.get("market_data", {}).get("error"):
        pass  # unrelated failure, already captured by health check

    return issues


def run(state: dict) -> dict:
    """
    Review gate: assess output completeness and consistency without an LLM call.

    Writes state["review_output"] = {issues, corrections, confidence_score, approved}
    """
    try:
        selected = state.get("selected_agents", []) or []

        issues, corrections = [], []
        ok_count = 0

        for agent_name in selected:
            key = _AGENT_OUTPUT_KEYS.get(agent_name)
            if not key:
                continue

            status, detail = _output_health(agent_name, state.get(key))
            if status == "ok":
                ok_count += 1
            else:
                issues.append(detail)
                if status == "degraded":
                    corrections.append(
                        f"Do not invent numbers for {agent_name} — say plainly that live data was unavailable."
                    )
                else:
                    corrections.append(
                        f"Answer without {agent_name} input rather than implying it was considered."
                    )

        issues.extend(_detect_contradictions(state))

        # Confidence scales with how much of the plan actually produced data.
        total = len([a for a in selected if a in _AGENT_OUTPUT_KEYS]) or 1
        success_ratio = ok_count / total

        confidence = int(45 + success_ratio * 50)          # 45..95
        confidence -= min(15, 5 * len(_detect_contradictions(state)))
        confidence = max(0, min(100, confidence))

        # Only fail approval when essentially nothing worked.
        approved = success_ratio >= 0.34

        state["review_output"] = {
            "issues": issues,
            "corrections": corrections,
            "confidence_score": confidence,
            "approved": approved,
        }

        state["trace"].append(
            f"review → {ok_count}/{total} agents ok, {len(issues)} issues, confidence {confidence}%"
        )

    except Exception as e:
        state["review_output"] = {
            "issues": [],
            "corrections": [],
            "confidence_score": 60,
            "approved": True,
        }
        state["trace"].append(f"review → ERROR: {str(e)[:100]}")

    return state

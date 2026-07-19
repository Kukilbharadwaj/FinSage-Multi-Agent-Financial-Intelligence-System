# config/models.py
# Groq model names used across all agents.
#
# NOTE: "qwen/qwen3-32b" was removed from Groq's catalog and now returns
# HTTP 404 (model_not_found). Every agent that used it (tax, technical,
# trading, mutual_fund) silently fell into its exception branch and
# returned a "could not be completed" placeholder. It is replaced by
# openai/gpt-oss-120b, which supports reasoning_format="hidden".
#
# Verify the catalog with:  client.models.list()

GROQ_FAST      = "llama-3.1-8b-instant"      # Supervisor, sentiment, extraction — must be fast
GROQ_STANDARD  = "llama-3.3-70b-versatile"   # Market summary, salary plan, synthesis
GROQ_REASONING = "openai/gpt-oss-120b"       # Tax math, technical analysis, trading strategy


# ── Pricing ───────────────────────────────────────────────────────────────
# USD per 1M tokens, as published on groq.com/pricing.
#
# Langfuse ships a price table for OpenAI/Anthropic/Bedrock but not for Groq,
# so a generation carrying only token counts still shows $0 and leaves the
# Model Costs and User Consumption panels empty. llm.py prices each call from
# this table and sends the result as cost_details.
#
# Update these if Groq changes its rates — nothing else needs to change.
GROQ_PRICING = {
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b":     {"input": 0.15, "output": 0.75},
}

_PER_MILLION = 1_000_000


def price_call(model: str, input_tokens: int, output_tokens: int) -> dict:
    """
    Return Langfuse cost_details in USD for one completion.

    Returns {} for a model missing from GROQ_PRICING, which makes the call show
    up with token usage but no cost rather than a wrong cost.
    """
    rates = GROQ_PRICING.get(model)
    if not rates:
        return {}

    input_cost = (input_tokens or 0) * rates["input"] / _PER_MILLION
    output_cost = (output_tokens or 0) * rates["output"] / _PER_MILLION

    return {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost,
    }

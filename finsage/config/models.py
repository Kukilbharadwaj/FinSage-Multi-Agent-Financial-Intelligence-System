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

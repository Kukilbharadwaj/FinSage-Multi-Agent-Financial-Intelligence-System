# config/models.py
# Groq model names used across all agents.
# Do not change these unless Groq deprecates a model.

GROQ_FAST      = "llama-3.1-8b-instant"      # Intent, extraction, sentiment — must be fast
GROQ_STANDARD  = "llama-3.3-70b-versatile"   # Market summary, salary plan, synthesis
GROQ_REASONING = "qwen/qwen3-32b"             # Tax math, technical analysis, risk calc

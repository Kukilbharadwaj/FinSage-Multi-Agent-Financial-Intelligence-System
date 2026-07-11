# guardrails/config.py
# Provider registration for NeMo Guardrails.
#
# Since we use the OpenAI-compatible endpoint for Groq (configured in
# config.yml with base_url), no custom provider registration is needed.
# This file ensures GROQ_API_KEY is available as an environment variable.

import os
from dotenv import load_dotenv

# Ensure .env is loaded so GROQ_API_KEY is available
load_dotenv()

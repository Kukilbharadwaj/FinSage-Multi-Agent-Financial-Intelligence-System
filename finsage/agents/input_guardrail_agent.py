# agents/input_guardrail_agent.py
# Input Guardrail — runs BEFORE the Supervisor Agent.
#
# Uses NVIDIA NeMo Guardrails to check user input for:
#   1. Prompt injection / jailbreak attempts
#   2. Off-topic queries (non-financial)
#   3. Toxic or harmful content
#
# If the query is blocked, sets state["input_safe"] = False
# and provides a reject reason. The graph will route to reject_response.

import os
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Lazy-loaded singleton for NeMo Rails instance ──
_rails_instance = None
_rails_load_error = None


def _get_rails():
    """
    Lazily load the NeMo Guardrails instance.
    We do this once and cache it to avoid reloading config on every request.
    """
    global _rails_instance, _rails_load_error

    if _rails_load_error is not None:
        return None

    if _rails_instance is not None:
        return _rails_instance

    try:
        from nemoguardrails import LLMRails, RailsConfig
        from dotenv import load_dotenv
        load_dotenv()

        # NeMo's OpenAI engine reads OPENAI_API_KEY and OPENAI_BASE_URL
        # Set them to Groq's values so NeMo connects to Groq
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            os.environ["OPENAI_API_KEY"] = groq_key
            os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

        # Resolve the guardrails config directory relative to this file
        # agents/ -> finsage/ -> guardrails/
        config_dir = Path(__file__).resolve().parent.parent / "guardrails"

        if not config_dir.exists():
            logger.error(f"Guardrails config directory not found: {config_dir}")
            _rails_load_error = "Config directory not found"
            return None

        config = RailsConfig.from_path(str(config_dir))
        _rails_instance = LLMRails(config)
        logger.info("NeMo Guardrails input rail loaded successfully")
        return _rails_instance

    except Exception as e:
        logger.error(f"Failed to load NeMo Guardrails: {e}")
        _rails_load_error = str(e)
        return None


def _run_async(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

try:
    from langfuse.decorators import observe
except ImportError:
    # Dummy decorator if langfuse is not installed
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@observe(as_type="generation")
def run(state: dict) -> dict:
    """
    Input Guardrail: check user query against NeMo Guardrails.

    Writes to state:
        - input_safe: True if the query passes all guardrails
        - input_reject_reason: reason string if blocked (empty if safe)
    """
    raw_query = state.get("raw_query", "").strip()

    # Empty queries are handled by the API layer, but guard here too
    if not raw_query:
        state["input_safe"] = False
        state["input_reject_reason"] = "Empty query received."
        state["trace"].append("input_guardrail → BLOCKED: empty query")
        return state

    rails = _get_rails()

    if rails is None:
        # If NeMo Guardrails failed to load, fail-open (allow the query)
        # This ensures the system still works even if guardrails have issues
        logger.warning("NeMo Guardrails not available — failing open (allowing query)")
        state["input_safe"] = True
        state["input_reject_reason"] = ""
        state["trace"].append(
            f"input_guardrail → WARN: NeMo not loaded ({_rails_load_error}), failing open"
        )
        return state

    try:
        # Run the NeMo Guardrails input check
        messages = [{"role": "user", "content": raw_query}]

        response = _run_async(rails.generate_async(messages=messages))

        # NeMo returns the response content — if the input was blocked,
        # the response will contain the refusal message from rails.co
        bot_message = ""
        if isinstance(response, dict):
            bot_message = response.get("content", "")
        elif isinstance(response, list):
            for msg in response:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    bot_message = msg.get("content", "")
                    break
        elif isinstance(response, str):
            bot_message = response

        # Check if the response is a refusal/block message
        refusal_indicators = [
            "i can only help with",
            "i cannot comply",
            "i'm sorry, i can only",
            "cannot help with that",
            "i am not able to help",
            "i can't respond to that",
        ]

        is_blocked = any(
            indicator in bot_message.lower()
            for indicator in refusal_indicators
        )

        if is_blocked:
            state["input_safe"] = False
            state["input_reject_reason"] = bot_message
            state["trace"].append(
                f"input_guardrail → BLOCKED: {bot_message[:80]}"
            )
        else:
            state["input_safe"] = True
            state["input_reject_reason"] = ""
            state["trace"].append("input_guardrail → PASSED")

    except Exception as e:
        # On error, fail-open to not break the system
        logger.error(f"Input guardrail error: {e}")
        state["input_safe"] = True
        state["input_reject_reason"] = ""
        state["trace"].append(
            f"input_guardrail → ERROR: {str(e)[:80]}, failing open"
        )

    return state

# agents/output_guardrail_agent.py
# Output Guardrail — runs AFTER the Synthesis Agent.
#
# Uses NVIDIA NeMo Guardrails to check the final recommendation for:
#   1. Guaranteed return claims ("100% profit", "guaranteed returns")
#   2. Missing disclaimers
#   3. Non-compliant financial advice language
#
# If the output is blocked, appends a strong disclaimer and flags it.
# If the output passes, ensures the SEBI disclaimer is present.

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard SEBI compliance disclaimer
SEBI_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Disclaimer:** This analysis is for educational and informational purposes only. "
    "FinSage AI is not a SEBI-registered investment advisor. Always consult a qualified "
    "financial advisor before making investment decisions. Past performance does not "
    "guarantee future results."
)

# Fallback response when output is blocked
COMPLIANCE_FALLBACK = (
    "I generated a response, but our compliance review flagged potential issues with it "
    "(such as overly confident claims or missing risk warnings). "
    "For your safety, here is general guidance:\n\n"
    "- **No investment is guaranteed** — all investments carry risk.\n"
    "- **Diversify** your portfolio across asset classes.\n"
    "- **Consult a SEBI-registered advisor** for personalized recommendations.\n"
    "- **Do your own research** before making any financial decision.\n\n"
    "Please rephrase your question for a more specific analysis, or consult a "
    "qualified financial professional."
)

# ── Lazy-loaded singleton for NeMo Rails instance ──
_rails_instance = None
_rails_load_error = None


def _get_rails():
    """Lazily load the NeMo Guardrails instance (shared with input guardrail)."""
    global _rails_instance, _rails_load_error

    if _rails_load_error is not None:
        return None

    if _rails_instance is not None:
        return _rails_instance

    try:
        from nemoguardrails import LLMRails, RailsConfig
        from dotenv import load_dotenv
        import os
        load_dotenv()

        # NeMo's OpenAI engine reads OPENAI_API_KEY and OPENAI_BASE_URL
        # Set them to Groq's values so NeMo connects to Groq
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            os.environ["OPENAI_API_KEY"] = groq_key
            os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

        config_dir = Path(__file__).resolve().parent.parent / "guardrails"

        if not config_dir.exists():
            logger.error(f"Guardrails config directory not found: {config_dir}")
            _rails_load_error = "Config directory not found"
            return None

        config = RailsConfig.from_path(str(config_dir))
        _rails_instance = LLMRails(config)
        logger.info("NeMo Guardrails output rail loaded successfully")
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
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _ensure_disclaimer(text: str) -> str:
    """Ensure the SEBI disclaimer is present at the end of the recommendation."""
    disclaimer_keywords = ["sebi", "not registered", "educational", "consult a qualified"]
    text_lower = text.lower()

    # Check if any disclaimer-like text already exists
    has_disclaimer = sum(1 for kw in disclaimer_keywords if kw in text_lower) >= 2

    if has_disclaimer:
        return text
    else:
        return text + SEBI_DISCLAIMER


def run(state: dict) -> dict:
    """
    Output Guardrail: check the final recommendation against NeMo Guardrails.

    Reads:
        - state["recommendation"]: the synthesis output

    Writes:
        - state["output_safe"]: True if the output passes compliance checks
        - state["recommendation"]: potentially modified with disclaimer or fallback
    """
    recommendation = state.get("recommendation", "")

    if not recommendation or not recommendation.strip():
        state["output_safe"] = True
        state["trace"].append("output_guardrail → SKIPPED: no recommendation to check")
        return state

    rails = _get_rails()

    if rails is None:
        # If NeMo Guardrails failed to load, still ensure the disclaimer is present
        logger.warning("NeMo Guardrails not available — ensuring disclaimer only")
        state["recommendation"] = _ensure_disclaimer(recommendation)
        state["output_safe"] = True
        state["trace"].append(
            f"output_guardrail → WARN: NeMo not loaded, disclaimer ensured"
        )
        return state

    try:
        # Run the NeMo output check
        # We send the recommendation as a bot message to check against output rails
        messages = [
            {"role": "user", "content": state.get("raw_query", "financial question")},
            {"role": "assistant", "content": recommendation},
        ]

        response = _run_async(rails.generate_async(messages=messages))

        # Extract the response
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

        # Check if NeMo replaced/blocked the output
        # If the output rail blocked it, the response will be different from what we sent
        refusal_indicators = [
            "i can't respond",
            "i cannot provide",
            "blocked",
            "i'm not able to",
            "cannot comply",
        ]

        is_blocked = any(
            indicator in bot_message.lower()
            for indicator in refusal_indicators
        )

        if is_blocked:
            # Output was blocked — replace with compliance fallback
            state["recommendation"] = COMPLIANCE_FALLBACK + SEBI_DISCLAIMER
            state["output_safe"] = False
            state["confidence"] = min(state.get("confidence", 0), 30)
            state["trace"].append("output_guardrail → BLOCKED: non-compliant output replaced")
        else:
            # Output passed — ensure disclaimer is present
            state["recommendation"] = _ensure_disclaimer(recommendation)
            state["output_safe"] = True
            state["trace"].append("output_guardrail → PASSED")

    except Exception as e:
        # On error, ensure disclaimer and continue
        logger.error(f"Output guardrail error: {e}")
        state["recommendation"] = _ensure_disclaimer(recommendation)
        state["output_safe"] = True
        state["trace"].append(
            f"output_guardrail → ERROR: {str(e)[:80]}, disclaimer ensured"
        )

    return state

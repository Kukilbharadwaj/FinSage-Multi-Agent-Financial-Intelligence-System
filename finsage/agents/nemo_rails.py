# agents/nemo_rails.py
# NVIDIA NeMo Guardrails runtime for FinSage.
#
# This module owns the NeMo layer: it loads guardrails/ (config.yml + rails.co),
# registers FinSage's policy as two custom NeMo actions, and exposes two sync
# helpers the LangGraph nodes call:
#
#   check_input(query, history)  -> {"action", "reason", "reply"}
#   check_output(text)           -> (cleaned_text, was_modified)
#
# Two deliberate choices:
#
# 1. The rails execute Python, not prompts. NeMo's built-in self_check_* rails
#    run a full LLM generation per check. The policy in agents/guardrail.py is
#    deterministic string matching, so wiring it in as a NeMo action keeps
#    NeMo's flow semantics (stop, rewrite, dialog state) at microsecond cost.
#
# 2. Every entry point degrades to the local policy. If nemoguardrails is not
#    installed or the config fails to load, check_input/check_output still work
#    by calling agents.guardrail directly. A guardrail layer that can take the
#    whole app down with it is worse than no guardrail layer.

import os
import threading

from agents.guardrail import classify, sanitize_output

# Path to the NeMo config directory (guardrails/ at the project root).
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guardrails")

# Built once, lazily, behind a lock — LLMRails construction parses Colang and is
# far too heavy to repeat per request, and stage agents run on multiple threads.
_rails = None
_rails_lock = threading.Lock()
_init_failed = False
_init_error = ""


def _build_rails():
    """Construct the LLMRails instance and register FinSage's policy actions."""
    from nemoguardrails import LLMRails, RailsConfig
    from nemoguardrails.actions import action

    @action(name="finsage_input_check")
    async def finsage_input_check(context: dict = None):
        """NeMo action: classify the incoming user message."""
        ctx = context or {}
        query = ctx.get("user_message", "") or ""
        # Seeded by check_input via a "context" role message (see below).
        history = ctx.get("conversation_history") or []
        return classify(query, history=history)

    @action(name="finsage_output_check")
    async def finsage_output_check(context: dict = None):
        """NeMo action: soften certainty language and ensure the disclaimer."""
        ctx = context or {}
        text = ctx.get("bot_message", "") or ""
        cleaned, modified = sanitize_output(text)
        return {"text": cleaned, "modified": modified}

    config = RailsConfig.from_path(_CONFIG_PATH)
    rails = LLMRails(config)
    rails.register_action(finsage_input_check, "finsage_input_check")
    rails.register_action(finsage_output_check, "finsage_output_check")
    return rails


def _get_rails():
    """Return the shared LLMRails instance, or None if NeMo is unavailable."""
    global _rails, _init_failed, _init_error

    if _rails is not None or _init_failed:
        return _rails

    with _rails_lock:
        if _rails is not None or _init_failed:
            return _rails
        try:
            _rails = _build_rails()
        except Exception as e:
            _init_failed = True
            _init_error = str(e)[:200]
            _rails = None

    return _rails


def is_active() -> bool:
    """True when the NeMo rails loaded successfully."""
    return _get_rails() is not None


def status() -> dict:
    """Diagnostic summary for the /health endpoint."""
    active = is_active()
    return {
        "engine": "nvidia-nemo-guardrails" if active else "local-fallback",
        "active": active,
        "config_path": _CONFIG_PATH,
        "input_rail": "finsage input policy",
        "output_rail": "finsage output policy",
        "llm_calls_per_check": 0,
        "error": _init_error or None,
    }


def check_input(query: str, history: list = None) -> dict:
    """
    Run the NeMo input rail.

    Returns {"action": "allow"|"smalltalk"|"block", "reason": str, "reply": str}.
    Falls back to the local policy if NeMo is unavailable or errors mid-flight.
    """
    rails = _get_rails()
    if rails is None:
        verdict = classify(query, history=history)
        verdict["engine"] = "local"
        return verdict

    try:
        from nemoguardrails.rails.llm.options import GenerationOptions

        result = rails.generate(
            messages=[
                # A "context" message seeds Colang context variables, which is
                # how conversation history reaches the action without a global.
                {"role": "context", "content": {"conversation_history": history or []}},
                {"role": "user", "content": query},
            ],
            options=GenerationOptions(
                # Input rails only — NeMo must not attempt its own generation,
                # the agent pipeline downstream produces the actual answer.
                rails=["input"],
                output_vars=["finsage_action", "finsage_reason", "finsage_reply"],
            ),
        )

        data = result.output_data or {}
        action_taken = data.get("finsage_action") or "allow"

        return {
            "action": action_taken,
            "reason": data.get("finsage_reason") or "passed",
            # On "allow" the rail echoes the user message back, so only trust
            # the reply variable the flow explicitly set.
            "reply": data.get("finsage_reply") or "",
            "engine": "nemo",
        }

    except Exception as e:
        verdict = classify(query, history=history)
        verdict["engine"] = f"local (nemo error: {str(e)[:80]})"
        return verdict


def check_output(text: str) -> tuple:
    """
    Run the NeMo output rail.

    Returns (cleaned_text, was_modified). Falls back to the local sanitizer.
    """
    if not text or not text.strip():
        return text, False

    rails = _get_rails()
    if rails is None:
        return sanitize_output(text)

    try:
        from nemoguardrails.rails.llm.options import GenerationOptions

        result = rails.generate(
            messages=[
                {"role": "user", "content": "finsage answer"},
                {"role": "assistant", "content": text},
            ],
            options=GenerationOptions(rails=["output"], output_vars=["finsage_output_modified"]),
        )

        response = result.response
        cleaned = response[0].get("content", text) if response else text
        modified = bool((result.output_data or {}).get("finsage_output_modified"))
        return cleaned, modified

    except Exception:
        return sanitize_output(text)

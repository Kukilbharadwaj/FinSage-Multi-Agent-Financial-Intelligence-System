# llm.py
# Instrumented, pooled Groq client — the single place agents get an LLM from.
#
# Why this module exists:
#
#   1. NO GENERATIONS IN LANGFUSE. Every agent called the raw `groq` SDK
#      directly. The Langfuse LangChain CallbackHandler only instruments
#      LangChain/LangGraph runnables, so it recorded the graph nodes (CHAIN)
#      and the MCP tools (TOOL) but never the model calls themselves. With
#      zero GENERATION observations Langfuse has no model name and no token
#      counts, which is why Model Usage, Model Costs and User Consumption
#      were all empty. This wrapper emits a proper generation per call.
#
#   2. NO COST. Langfuse only auto-prices models it knows, and Groq's catalog
#      is not in its default table. Cost is computed here from GROQ_PRICING
#      and sent explicitly as cost_details, so the cost panels populate.
#
#   3. A NEW CLIENT PER CALL. `Groq(api_key=...)` was constructed inside each
#      agent's run(), so all ~5 model calls in a request paid a fresh TLS
#      handshake. Clients are cached per API key here and reused.
#
#   4. NO TIMEOUT. Groq calls had no deadline, so one hung request stalled the
#      agent until the 45s stage timeout fired — that is the p99 you saw.
#
# Drop-in usage — agents only change their import:
#
#     from llm import Groq          # was: from groq import Groq
#
# The returned object exposes the same `.chat.completions.create(...)` call,
# so no other agent code changes.

import logging
import os
import time

from config.models import price_call

logger = logging.getLogger(__name__)

# A single model call should never outlive the stage that owns it. Groq is
# fast; anything past this is a stall, and failing at 20s lets the agent fall
# into its own exception branch while the rest of the stage still completes.
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))

# Transient 5xx/connection blips are common enough to be worth one retry, but
# retries multiply latency, so the SDK default (2) is lowered.
DEFAULT_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "1"))

_clients: dict = {}


def _raw_client(api_key: str):
    """Return a cached groq.Groq for this key, building it on first use."""
    cached = _clients.get(api_key)
    if cached is not None:
        return cached

    from groq import Groq as _RawGroq

    client = _RawGroq(
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    _clients[api_key] = client
    return client


def _messages_as_input(messages) -> list:
    """Shape messages for the Langfuse input panel, trimming huge prompts."""
    shaped = []
    for message in messages or []:
        content = message.get("content", "")
        if isinstance(content, str) and len(content) > 8000:
            content = content[:8000] + f"\n…[truncated, {len(content)} chars total]"
        shaped.append({"role": message.get("role", "user"), "content": content})
    return shaped


def _usage_from_response(response) -> dict:
    """Extract Langfuse-shaped token counts from a Groq response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or (prompt + completion)

    # Langfuse v3+ expects the generic input/output/total keys.
    return {"input": prompt, "output": completion, "total": total}


class _Completions:
    """Mirrors groq.resources.chat.Completions, adding Langfuse tracing."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    def create(self, *, model: str, messages: list, name: str = "", **kwargs):
        """
        Call Groq and record the result as a Langfuse generation.

        `name` is an optional label for the Langfuse observation. It is popped
        before the request so it never reaches the Groq API.
        """
        client = _raw_client(self._api_key)
        observation_name = name or f"groq:{model}"

        generation = _start_generation(
            name=observation_name,
            model=model,
            messages=messages,
            params=kwargs,
        )

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except Exception as exc:
            _end_generation(
                generation,
                model=model,
                output=None,
                usage={},
                error=exc,
                elapsed=time.perf_counter() - started,
            )
            raise

        _end_generation(
            generation,
            model=model,
            output=_first_message(response),
            usage=_usage_from_response(response),
            error=None,
            elapsed=time.perf_counter() - started,
        )
        return response


def _first_message(response) -> str:
    """Best-effort extraction of the assistant text from a Groq response."""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _start_generation(*, name: str, model: str, messages: list, params: dict):
    """
    Open a Langfuse generation. Returns None when telemetry is off, and every
    call site tolerates None — tracing must never break a user request.
    """
    from observability import is_enabled

    if not is_enabled():
        return None

    try:
        from langfuse import get_client

        # v4 exposes one entry point, start_observation(as_type=...); there is
        # no start_generation. as_type="generation" is what makes Langfuse
        # treat this as a model call and surface it under Model Usage/Costs.
        return get_client().start_observation(
            name=name,
            as_type="generation",
            model=model,
            input=_messages_as_input(messages),
            model_parameters={
                key: value
                for key, value in params.items()
                if key in ("temperature", "max_tokens", "top_p", "reasoning_format")
            },
        )
    except Exception as exc:
        logger.debug("[Langfuse] could not start generation: %s", exc)
        return None


def _end_generation(generation, *, model, output, usage, error, elapsed) -> None:
    """Attach output, token usage and computed cost, then close the span."""
    if generation is None:
        return

    try:
        payload = {
            "output": output if error is None else None,
            "metadata": {"latency_seconds": round(elapsed, 3)},
        }

        if usage:
            payload["usage_details"] = usage
            # Groq is absent from Langfuse's built-in price table, so the cost
            # panels stay empty unless we price the call ourselves.
            cost = price_call(model, usage["input"], usage["output"])
            if cost:
                payload["cost_details"] = cost

        if error is not None:
            payload["level"] = "ERROR"
            payload["status_message"] = str(error)[:500]
        elif not (output or "").strip():
            # A reasoning model whose hidden chain-of-thought consumes the whole
            # max_tokens budget returns HTTP 200 with empty content. The agent
            # then .strip()s it into an empty analysis and the pipeline carries
            # on with nothing — invisible unless it is flagged here.
            payload["level"] = "WARNING"
            payload["status_message"] = (
                "Empty completion — likely max_tokens exhausted by hidden "
                "reasoning before any visible text was produced."
            )

        generation.update(**payload)
        generation.end()
    except Exception as exc:
        logger.debug("[Langfuse] could not close generation: %s", exc)


class _Chat:
    def __init__(self, api_key: str):
        self.completions = _Completions(api_key)


class Groq:
    """
    Drop-in replacement for groq.Groq that traces every call to Langfuse.

    The underlying HTTP client is shared per API key, so constructing this
    object inside an agent's run() is cheap and does not open a new connection.
    """

    def __init__(self, api_key: str = "", **_ignored):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.chat = _Chat(self.api_key)


__all__ = ["Groq", "DEFAULT_TIMEOUT_SECONDS"]

# observability.py
# Langfuse telemetry wiring — single place that knows how to talk to Langfuse.
#
# Why the dashboard was empty before this existed:
#
#   1. WRONG IMPORT PATHS. The code imported `langfuse.callback.CallbackHandler`
#      and `langfuse.decorators.observe`. Those modules were removed in Langfuse
#      v3; this project runs v4, where they live at `langfuse.langchain
#      .CallbackHandler` and `langfuse.observe`. Every import raised
#      ImportError, and every call site caught it and silently substituted a
#      no-op decorator / an empty callback list. So nothing was ever traced —
#      the failure was completely invisible.
#
#   2. WRONG ENV VAR. .env defines LANGFUSE_BASE_URL, but the SDK reads
#      LANGFUSE_HOST. Without it the client defaults to cloud.langfuse.com
#      while these keys belong to us.cloud.langfuse.com, so even a correct
#      handler would have authenticated against the wrong region.
#
# Both are handled here, and failures are logged rather than swallowed.

#   3. NO GENERATIONS, SO NO COST OR USAGE. Every agent called the raw `groq`
#      SDK, which the LangChain callback handler cannot see — it only observes
#      LangChain/LangGraph runnables. Traces therefore contained CHAIN and TOOL
#      observations but not a single GENERATION, leaving Model Usage, Model
#      Costs and User Consumption permanently empty. Model calls now go through
#      llm.py, which emits a priced generation per call.
#
#   4. NO SCORES. Nothing ever wrote one. The pipeline already computes a
#      review-gate confidence and an approval flag; `score()` below ships them
#      to Langfuse so the Scores panel reflects answer quality.

import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_client = None
_enabled = False
_initialised = False


def _normalise_env() -> None:
    """Accept LANGFUSE_BASE_URL as an alias for the SDK's LANGFUSE_HOST."""
    if not os.environ.get("LANGFUSE_HOST"):
        base_url = os.environ.get("LANGFUSE_BASE_URL", "").strip()
        if base_url:
            os.environ["LANGFUSE_HOST"] = base_url


def init_langfuse() -> bool:
    """
    Initialise the Langfuse client once and verify credentials.

    Returns True when tracing is live. Safe to call repeatedly.
    """
    global _client, _enabled, _initialised

    if _initialised:
        return _enabled
    _initialised = True

    _normalise_env()

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()

    if not (public_key and secret_key):
        logger.info("[Langfuse] Keys not set - telemetry disabled")
        return False

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        # auth_check() makes a real request, so a bad key surfaces at startup
        # instead of silently producing zero traces.
        if not _client.auth_check():
            logger.warning("[Langfuse] Authentication failed - check your keys and host")
            _client = None
            return False

        _enabled = True
        logger.info("[Langfuse] Connected to %s", os.environ.get("LANGFUSE_HOST"))
        return True

    except Exception as exc:
        logger.warning("[Langfuse] Initialisation failed: %s", exc)
        _client = None
        return False


def is_enabled() -> bool:
    """Return whether Langfuse tracing is active."""
    return _enabled


def get_callbacks(user_id: str = "", session_id: str = "", tags: list = None) -> list:
    """
    Build the LangChain callback list for one graph run.

    Returns [] when telemetry is off, so callers can pass the result straight
    into graph.invoke(config={"callbacks": ...}) either way.
    """
    if not init_langfuse():
        return []

    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as exc:
        logger.warning("[Langfuse] Could not build callback handler: %s", exc)
        return []


@contextmanager
def trace(name: str, user_id: str = "", session_id: str = "", tags: list = None,
          input: object = None):
    """
    Open a root span for one request and yield its trace_id (None when off).

    Two things make the dashboards work, and both need this wrapper:

      - propagate_attributes() stamps user_id/session_id onto this span AND
        every child span created inside the context. Langfuse's aggregation
        queries only count observations that carry the attribute, so setting
        it on the root alone leaves User Consumption empty — the per-generation
        costs simply are not attributed to anyone. It must be entered before
        the graph runs; spans opened earlier are not backfilled.

      - The yielded trace_id lets scores be attached after the run, once this
        span context has already closed.

    NOTE for anyone maintaining this: v4 renamed these APIs. There is no
    `update_current_trace` and no `start_as_current_span` on the client — it is
    `start_as_current_observation(as_type=...)` plus `propagate_attributes`.
    The old names fail silently into the except branch.
    """
    if not init_langfuse():
        yield None
        return

    try:
        from langfuse import get_client, propagate_attributes

        client = get_client()
        with client.start_as_current_observation(
            name=name, as_type="span", input=input
        ) as span:
            with propagate_attributes(
                user_id=user_id or None,
                session_id=session_id or None,
                tags=tags or None,
                trace_name=name,
            ):
                yield span.trace_id
    except Exception as exc:
        logger.warning("[Langfuse] trace span failed: %s", exc)
        yield None


def score(trace_id: str, name: str, value, comment: str = "",
          data_type: str = "NUMERIC") -> None:
    """
    Attach a score to a finished trace.

    trace_id is passed explicitly rather than relying on the ambient context,
    because scoring happens after the graph run has already exited its span.
    """
    if not _enabled or not trace_id:
        return

    try:
        from langfuse import get_client

        get_client().create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type=data_type,
            comment=comment[:500] if comment else None,
        )
    except Exception as exc:
        logger.debug("[Langfuse] score '%s' skipped: %s", name, exc)


def update_trace(**kwargs) -> None:
    """Attach metadata (user_id, session_id, tags, input/output) to the active trace."""
    if not _enabled:
        return
    try:
        from langfuse import get_client

        get_client().update_current_trace(**kwargs)
    except Exception as exc:
        logger.debug("[Langfuse] update_trace skipped: %s", exc)


def flush() -> None:
    """Flush buffered events — call on shutdown so nothing is lost."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            pass


def observe(*args, **kwargs):
    """
    Re-export Langfuse's @observe decorator, degrading to a no-op when the SDK
    is unavailable. Import this instead of reaching into langfuse internals —
    that is what produced the silent-failure bug described at the top.
    """
    try:
        from langfuse import observe as _observe

        return _observe(*args, **kwargs)
    except Exception:
        if args and callable(args[0]):
            return args[0]

        def _decorator(func):
            return func

        return _decorator

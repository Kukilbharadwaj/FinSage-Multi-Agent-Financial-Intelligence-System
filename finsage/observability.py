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

import logging
import os

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

# agents/memory.py
# Conversational (working) memory — the last N turns of a session.
#
# The frontend caps a session at 5 queries and sends a stable `user_id` for the
# whole session, so memory is keyed on user_id and holds at most MAX_TURNS
# turns. Every one of those turns is handed to the agents on each request,
# which is what lets follow-ups work: "and if I invest 20k instead?" only means
# something next to the question before it.
#
# Storage is a bounded in-process dict. That matches the session's lifetime and
# needs no schema change; when the process restarts mid-session, `hydrate` pulls
# the turns back out of the query_logs table that routes.py already writes.

import threading
from collections import OrderedDict, deque

# Matches MAX_QUERIES in frontend/static/app.js — a session is 5 messages, and
# all 5 stay in memory, so the last turn still sees the first.
MAX_TURNS = 5

# Ceiling on concurrent sessions held in memory. Without it, a long-running
# process accumulates one entry per visitor forever.
_MAX_SESSIONS = 500

_sessions = OrderedDict()   # user_id -> deque[dict]
_lock = threading.Lock()


def get_history(user_id: str) -> list:
    """Return this session's turns, oldest first. Empty list when unknown."""
    if not user_id:
        return []

    with _lock:
        turns = _sessions.get(user_id)
        if turns is None:
            return []
        # Refresh recency so an active session is not evicted mid-conversation.
        _sessions.move_to_end(user_id)
        return list(turns)


def remember(user_id: str, query: str, answer: str, intent: str = "") -> None:
    """Append one completed turn, evicting the oldest beyond MAX_TURNS."""
    if not user_id or not query:
        return

    turn = {
        "query": query.strip(),
        # Answers can run long; memory only needs enough to resolve references.
        "answer": (answer or "").strip()[:1200],
        "intent": intent or "",
    }

    with _lock:
        turns = _sessions.get(user_id)
        if turns is None:
            turns = deque(maxlen=MAX_TURNS)
            _sessions[user_id] = turns

        turns.append(turn)
        _sessions.move_to_end(user_id)

        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)


def hydrate(user_id: str, rows: list) -> list:
    """
    Seed memory from persisted query_logs rows (newest first, as crud returns).

    Only used when the in-process session is empty — a restart should not cost
    the user the context of the session they are in the middle of.
    """
    if not user_id or not rows:
        return []

    with _lock:
        if _sessions.get(user_id):
            return list(_sessions[user_id])

        turns = deque(maxlen=MAX_TURNS)
        for row in reversed(rows[:MAX_TURNS]):   # oldest first
            turns.append({
                "query": (getattr(row, "raw_query", "") or "").strip(),
                "answer": (getattr(row, "recommendation", "") or "").strip()[:1200],
                "intent": getattr(row, "intent", "") or "",
            })

        _sessions[user_id] = turns
        _sessions.move_to_end(user_id)
        return list(turns)


def clear(user_id: str) -> None:
    """Drop a session's memory (new session / explicit reset)."""
    with _lock:
        _sessions.pop(user_id, None)


def format_history(history: list, answer_chars: int = 400) -> str:
    """
    Render turns as a transcript for an LLM prompt.

    Answers are truncated because the agents need the thread of the
    conversation, not a verbatim replay of everything already said.
    """
    if not history:
        return ""

    lines = []
    for i, turn in enumerate(history, 1):
        query = (turn.get("query") or "").strip()
        answer = (turn.get("answer") or "").strip()
        if not query:
            continue

        lines.append(f"[Turn {i}] User: {query}")
        if answer:
            snippet = answer[:answer_chars]
            if len(answer) > answer_chars:
                snippet += " …"
            lines.append(f"[Turn {i}] FinSage: {snippet}")

    return "\n".join(lines)

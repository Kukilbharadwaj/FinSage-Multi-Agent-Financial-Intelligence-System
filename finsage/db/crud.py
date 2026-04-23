# db/crud.py
# Database CRUD operations for query logging.

from sqlalchemy.orm import Session
from db.models import QueryLog


def save_query_log(
    db: Session,
    user_id: str,
    raw_query: str,
    intent: str,
    recommendation: str = None,
    confidence: int = None,
) -> QueryLog:
    """Save a query log entry to the database."""
    log = QueryLog(
        user_id=user_id,
        raw_query=raw_query,
        intent=intent,
        recommendation=recommendation,
        confidence=confidence,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_recent_queries(db: Session, user_id: str, limit: int = 5) -> list:
    """Return the last N queries for a user, ordered by most recent first."""
    return (
        db.query(QueryLog)
        .filter(QueryLog.user_id == user_id)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )

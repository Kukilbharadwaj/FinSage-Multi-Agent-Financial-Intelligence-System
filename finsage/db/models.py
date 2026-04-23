# db/models.py
# SQLAlchemy table models for query logging.

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime
from db.database import Base


class QueryLog(Base):
    """Stores every user query and the system's response."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), index=True, nullable=False)
    raw_query = Column(Text, nullable=False)
    intent = Column(String(50), nullable=True)
    recommendation = Column(Text, nullable=True)
    confidence = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<QueryLog(id={self.id}, user='{self.user_id}', intent='{self.intent}')>"

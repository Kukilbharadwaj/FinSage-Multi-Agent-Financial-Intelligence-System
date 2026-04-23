# api/routes.py
# FastAPI router with /chat, /health, and /history endpoints.

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from agents.graph import app_graph
from db.database import get_db
from db.crud import save_query_log, get_recent_queries

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    user_id: str = Field(default="default_user", description="User session identifier")
    query: str = Field(..., description="The financial question to analyze")


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    answer: str
    confidence: int
    intent: str
    trace: list


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Process a financial query through the FinSage AI agent graph.

    Runs intent detection, fetches data, and returns a recommendation.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Build initial AgentState with all fields set to zero/None/empty values
        initial_state = {
            "user_id": request.user_id,
            "raw_query": request.query.strip(),
            "intent": "",
            "entities": {},
            "market_data": None,
            "ohlcv": None,
            "news": None,
            "rag_context": None,
            "technical_signals": None,
            "sentiment_score": None,
            "salary_plan": None,
            "tax_result": None,
            "recommendation": None,
            "confidence": None,
            "data_freshness": datetime.now(timezone.utc).isoformat(),
            "trace": [],
        }

        # Run the full agent graph
        result = app_graph.invoke(initial_state)

        # Extract results
        recommendation = result.get("recommendation", "No recommendation generated.")
        confidence = result.get("confidence", 0)
        intent = result.get("intent", "general")
        trace = result.get("trace", [])

        # Save to database
        try:
            save_query_log(
                db=db,
                user_id=request.user_id,
                raw_query=request.query,
                intent=intent,
                recommendation=recommendation,
                confidence=confidence,
            )
        except Exception:
            pass  # Don't fail the request if DB save fails

        return ChatResponse(
            answer=recommendation,
            confidence=confidence or 0,
            intent=intent,
            trace=trace,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your query: {str(e)[:300]}",
        )


@router.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@router.get("/history/{user_id}")
def history(user_id: str, db: Session = Depends(get_db)):
    """Return the last 5 queries for a user."""
    try:
        queries = get_recent_queries(db, user_id, limit=5)
        return {
            "user_id": user_id,
            "queries": [
                {
                    "id": q.id,
                    "query": q.raw_query,
                    "intent": q.intent,
                    "confidence": q.confidence,
                    "created_at": q.created_at.isoformat() if q.created_at else None,
                }
                for q in queries
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)[:200]}")

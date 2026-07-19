# api/routes.py
# FastAPI router with /chat, /health, and /history endpoints.

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from agents.graph import app_graph
from agents import memory
from db.database import get_db
from db.crud import save_query_log, get_recent_queries
from mcp_bridge import has_live_session, get_tools
from observability import get_callbacks, is_enabled as langfuse_enabled, score, trace

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

    Runs supervisor planning, staged agent execution, review, and synthesis.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # ── Working memory: every earlier turn of this session ──
        # A session is 5 messages and all 5 are replayed, so the last question
        # can still refer back to the first. On a cold process the turns are
        # rebuilt from query_logs rather than lost.
        history = memory.get_history(request.user_id)
        if not history:
            try:
                history = memory.hydrate(
                    request.user_id,
                    get_recent_queries(db, request.user_id, limit=memory.MAX_TURNS),
                )
            except Exception:
                history = []   # memory is an enhancement, never a hard dependency

        # Build initial FinSageState with all fields set to zero/None/empty values
        initial_state = {
            # Identity
            "user_id": request.user_id,
            "raw_query": request.query.strip(),

            # Working memory (populated above)
            "conversation_history": history,

            # Supervisor outputs (populated by supervisor_agent)
            "goal": "",
            "intent": "",
            "entities": {},
            "selected_agents": [],
            "execution_plan": [],

            # Communication bus: structured analysis dicts (populated by agents)
            "salary_analysis": None,
            "news_analysis": None,
            "general_finance_result": None,
            "tax_analysis": None,
            "market_analysis": None,
            "mf_analysis": None,
            "trading_analysis_output": None,
            "technical_analysis": None,

            # RAG context (populated on-demand by agents via rag_agent)
            "rag_context": None,

            # Review gate (populated by review_agent)
            "review_output": None,

            # Guardrail gate (populated by the local guardrail in agents/guardrail.py)
            "input_safe": None,
            "guardrail_action": None,
            "input_reject_reason": None,
            "output_safe": None,

            # Final output (populated by synthesis_agent)
            "recommendation": None,
            "confidence": None,
            "data_freshness": datetime.now(timezone.utc).isoformat(),
            "trace": [],
        }

        # Set up Langfuse telemetry callback (empty list when disabled)
        callbacks = get_callbacks()

        # Own the root span so user_id/session_id land on the trace itself and
        # so there is a trace_id to hang scores off once the run completes.
        with trace(
            name="finsage_query",
            user_id=request.user_id,
            session_id=request.user_id,
            tags=["finsage", "chat"],
            input=request.query.strip(),
        ) as trace_id:
            started = time.perf_counter()

            # Run the full agent graph
            result = app_graph.invoke(
                initial_state,
                config={
                    "callbacks": callbacks,
                    "run_name": "finsage_graph",
                    "metadata": {
                        "langfuse_user_id": request.user_id,
                        "langfuse_session_id": request.user_id,
                        "langfuse_tags": ["finsage", "chat"],
                    },
                },
            )

            elapsed = time.perf_counter() - started

        # Extract results
        recommendation = result.get("recommendation", "No recommendation generated.")
        confidence = result.get("confidence", 0)
        intent = result.get("intent", "general")
        agent_trace = result.get("trace", [])

        # ── Scores: the pipeline already grades itself, so publish it ──
        # Langfuse expects numeric scores in 0–1, hence the /100.
        review = result.get("review_output") or {}
        score(trace_id, "confidence", (confidence or 0) / 100,
              comment=f"intent={intent}")
        score(trace_id, "review_approved", 1 if review.get("approved", True) else 0,
              comment="; ".join(review.get("issues", [])[:3]))
        score(trace_id, "latency_seconds", round(elapsed, 3),
              comment=f"agents={','.join(result.get('selected_agents', []))}")

        # Add execution plan to trace for transparency
        plan = result.get("execution_plan", [])
        if plan:
            agent_trace = [f"📋 Plan: {' → '.join(plan[:6])}"] + agent_trace

        # Record the turn so the next message in this session can refer to it.
        memory.remember(
            user_id=request.user_id,
            query=request.query.strip(),
            answer=recommendation,
            intent=intent,
        )

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
            trace=agent_trace,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your query: {str(e)[:300]}",
        )


@router.get("/health")
def health():
    """Health check endpoint."""
    # Report the vector store inline — RAG now depends on two external
    # services, so a silent outage should be visible here rather than only
    # showing up as vague answers.
    try:
        from rag.embedder import EMBEDDING_DIM, MODEL_ID
        from rag.vector_store import INDEX_NAME, is_ready, vector_count

        rag_status = {
            "vector_store": "pinecone",
            "index": INDEX_NAME,
            "connected": is_ready(),
            "vectors": vector_count(),
            "embedding_model": MODEL_ID,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_source": "huggingface-api",
        }
    except Exception as exc:
        rag_status = {"vector_store": "pinecone", "connected": False, "error": str(exc)[:150]}

    # Surface which guardrail engine is live. If NeMo failed to load the app
    # still runs on the local fallback, and that should be visible here rather
    # than only inferrable from the trace.
    try:
        from agents.nemo_rails import status as guardrails_status
        rails_status = guardrails_status()
    except Exception as exc:
        rails_status = {"engine": "unknown", "active": False, "error": str(exc)[:150]}

    return {
        "status": "ok",
        "version": "0.5.0",
        "guardrails": rails_status,
        "conversation_memory": {"max_turns": memory.MAX_TURNS, "scope": "per user_id session"},
        "architecture": "supervisor-staged-parallel",
        "mcp_transport": "in-memory (fastmcp)",
        "mcp_connected": has_live_session(),
        "mcp_tools": get_tools(),
        "langfuse_enabled": langfuse_enabled(),
        "rag": rag_status,
    }


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

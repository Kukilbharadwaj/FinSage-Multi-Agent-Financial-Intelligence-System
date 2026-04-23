# agents/state.py
# AgentState TypedDict — the single shared data structure that flows through
# every node in the LangGraph graph. Every field must have a type annotation.

from typing import TypedDict, Optional


class AgentState(TypedDict):
    """Shared state flowing through all LangGraph nodes."""

    # User and query identification
    user_id: str                           # identifies the user session
    raw_query: str                         # the original user question unchanged

    # Intent classification results
    intent: str                            # one of: salary, stock, index, tax, general
    entities: dict                         # extracted: {stock, amount, index} — may be empty

    # Market data
    market_data: Optional[dict]            # price, change, high, low, 52w high/low, source
    ohlcv: Optional[dict]                  # close, high, low, volume, dates lists for charts

    # News and sentiment
    news: Optional[list]                   # list of {title, source, link} dicts
    rag_context: Optional[str]            # retrieved text chunks from FAISS index

    # Analysis results
    technical_signals: Optional[dict]      # ema20, ema50, rsi, macd, support, resistance, trend
    sentiment_score: Optional[float]       # -1.0 (very negative) to 1.0 (very positive)
    salary_plan: Optional[dict]            # {plan: str} with monthly breakdown
    tax_result: Optional[str]              # tax calculation narrative from qwq reasoning

    # Final output
    recommendation: Optional[str]          # the final formatted answer shown to user
    confidence: Optional[int]              # 0-100 confidence estimate
    data_freshness: str                    # ISO timestamp of when market data was fetched
    trace: list                            # list of strings showing which agents ran

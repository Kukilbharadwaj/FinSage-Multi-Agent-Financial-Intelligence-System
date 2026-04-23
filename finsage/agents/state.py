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
    intent: str                            # one of: salary, stock, index, tax, trading, mutual_fund, insurance, loan, retirement, gold, crypto, general
    entities: dict                         # extracted: {stock, amount, index, fund_name} — may be empty

    # Market data
    market_data: Optional[dict]            # price, change, high, low, 52w high/low, source
    company_profile: Optional[dict]        # fundamental data: P/E, EPS, market cap, sector, etc.
    ohlcv: Optional[dict]                  # close, high, low, volume, dates lists for charts
    intraday_data: Optional[dict]          # 5-min candle data for intraday analysis

    # Options / F&O data
    options_chain: Optional[dict]          # calls, puts, PCR, max pain, OI data

    # Mutual fund data
    mutual_fund_data: Optional[dict]       # NAV, returns, fund house, category

    # News and sentiment
    news: Optional[list]                   # list of {title, source, link} dicts
    rag_context: Optional[str]            # retrieved text chunks from FAISS index

    # Analysis results
    technical_signals: Optional[dict]      # ema20, ema50, rsi, macd, support, resistance, trend
    sentiment_score: Optional[float]       # -1.0 (very negative) to 1.0 (very positive)
    salary_plan: Optional[dict]            # {plan: str} with monthly breakdown
    tax_result: Optional[str]              # tax calculation narrative from qwq reasoning
    trading_analysis: Optional[str]        # intraday/options trading analysis
    general_finance_result: Optional[str]  # insurance, loan, retirement, gold, crypto answers

    # Final output
    recommendation: Optional[str]          # the final formatted answer shown to user
    confidence: Optional[int]              # 0-100 confidence estimate
    data_freshness: str                    # ISO timestamp of when market data was fetched
    trace: list                            # list of strings showing which agents ran

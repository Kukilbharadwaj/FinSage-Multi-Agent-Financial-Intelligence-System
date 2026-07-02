# agents/state.py
# FinSageState TypedDict — the shared data structure that flows through
# every node in the LangGraph graph.
#
# This state acts as a COMMUNICATION BUS:
#   - Each agent writes to its own *_analysis dict
#   - Downstream agents READ from upstream *_analysis dicts
#   - No direct agent-to-agent calls — all communication via state
#
# Dependency map (read direction):
#   salary_analysis  ← read by: tax_agent, mutual_fund_agent
#   tax_analysis     ← read by: mutual_fund_agent
#   news_analysis    ← read by: market_agent
#   market_analysis  ← read by: trading_agent, mutual_fund_agent

from typing import TypedDict, Optional


class FinSageState(TypedDict):
    """Shared state flowing through all LangGraph nodes."""

    # ── Identity ──────────────────────────────────────────────
    user_id: str                            # identifies the user session
    raw_query: str                          # the original user question unchanged

    # ── Supervisor outputs ────────────────────────────────────
    goal: str                               # what the user wants (human-readable)
    intent: str                             # primary intent for backward compat with API
    entities: dict                          # extracted: {stock, amount, index, fund_name}
    selected_agents: list                   # e.g. ["salary", "tax", "mutual_fund", "market"]
    execution_plan: list                    # e.g. ["Analyze salary", "Calculate tax", ...]

    # ── Communication bus: structured analysis dicts ──────────
    #
    # Stage 1 outputs (no dependencies):
    salary_analysis: Optional[dict]         # {annual_salary, monthly_salary, monthly_savings,
                                            #  investable_income, risk_profile, plan}
    news_analysis: Optional[dict]           # {headlines, sentiment_score, key_events,
                                            #  market_mood}
    general_finance_result: Optional[dict]  # {answer, topic}

    # Stage 2 outputs (read Stage 1):
    tax_analysis: Optional[dict]            # {tax_result, remaining_80c,
                                            #  tax_saving_opportunities, effective_rate}
    market_analysis: Optional[dict]         # {market_data, company_profile, sentiment,
                                            #  volatility, timing_recommendation, summary}

    # Stage 3 outputs (read Stage 1+2):
    mf_analysis: Optional[dict]             # {mutual_fund_data, sip_recommendation, analysis}
    trading_analysis_output: Optional[dict] # {analysis, market_status, options_data}
    technical_analysis: Optional[dict]      # {signals, ohlcv, analysis}

    # ── RAG context (populated on-demand by agents via rag_agent) ──
    rag_context: Optional[dict]             # {contexts: {tax: ..., salary: ...},
                                            #  sources: [...], combined: "..."}

    # ── Review gate ───────────────────────────────────────────
    review_output: Optional[dict]           # {issues, corrections, confidence_score, approved}

    # ── Final output ──────────────────────────────────────────
    recommendation: Optional[str]           # the final formatted answer shown to user
    confidence: Optional[int]               # 0-100 confidence estimate
    data_freshness: str                     # ISO timestamp of when data was fetched
    trace: list                             # list of strings showing which agents ran

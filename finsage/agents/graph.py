# agents/graph.py
# LangGraph StateGraph definition — the core orchestration pipeline.
#
# Architecture:
#   supervisor → stage_1 → stage_2 → stage_3 → review → synthesis → END
#
# Each stage is a dispatcher that only runs agents selected by the Supervisor
# AND assigned to that stage. Unselected stages are no-op passthroughs.
#
# Stage assignment (dependency-aware):
#   Stage 1 (independent):  salary, news, general_finance
#   Stage 2 (reads Stage 1): tax, market
#   Stage 3 (reads Stage 1+2): mutual_fund, trading, technical
#   Stage 4 (always): review
#   Stage 5 (always): synthesis

from langgraph.graph import StateGraph, END
from agents.state import FinSageState
import agents.supervisor_agent as supervisor_agent
import agents.salary_agent as salary_agent
import agents.news_agent as news_agent
import agents.general_finance_agent as general_finance_agent
import agents.tax_agent as tax_agent
import agents.market_agent as market_agent
import agents.mutual_fund_agent as mutual_fund_agent
import agents.trading_agent as trading_agent
import agents.technical_agent as technical_agent
import agents.review_agent as review_agent
import agents.synthesis_agent as synthesis_agent


# ── Agent registry: maps agent names to their run() functions ──
AGENT_REGISTRY = {
    "salary": salary_agent.run,
    "news": news_agent.run,
    "general_finance": general_finance_agent.run,
    "tax": tax_agent.run,
    "market": market_agent.run,
    "mutual_fund": mutual_fund_agent.run,
    "trading": trading_agent.run,
    "technical": technical_agent.run,
}

# ── Stage assignments: which stage each agent belongs to ──
STAGE_ASSIGNMENT = {
    # Stage 1: No dependencies — can run independently
    "salary": 1,
    "news": 1,
    "general_finance": 1,
    # Stage 2: Reads Stage 1 outputs
    "tax": 2,        # reads salary_analysis
    "market": 2,     # reads news_analysis
    # Stage 3: Reads Stage 1+2 outputs
    "mutual_fund": 3,  # reads salary_analysis + tax_analysis + market_analysis
    "trading": 3,      # reads market_analysis
    "technical": 3,    # independent (pure calculation)
}


def _run_stage(state: dict, stage_num: int) -> dict:
    """
    Run all selected agents assigned to this stage.
    If no agents are selected for this stage, this is a no-op passthrough.
    """
    selected = state.get("selected_agents", [])

    for agent_name in selected:
        if STAGE_ASSIGNMENT.get(agent_name) == stage_num:
            agent_fn = AGENT_REGISTRY.get(agent_name)
            if agent_fn:
                try:
                    state = agent_fn(state)
                except Exception as e:
                    state["trace"].append(
                        f"stage_{stage_num} → ERROR running {agent_name}: {str(e)[:100]}"
                    )

    return state


def run_stage_1(state: dict) -> dict:
    """Stage 1: Independent agents (salary, news, general_finance)."""
    return _run_stage(state, 1)


def run_stage_2(state: dict) -> dict:
    """Stage 2: Agents that read Stage 1 outputs (tax, market)."""
    return _run_stage(state, 2)


def run_stage_3(state: dict) -> dict:
    """Stage 3: Agents that read Stage 1+2 outputs (mutual_fund, trading, technical)."""
    return _run_stage(state, 3)


def build_graph() -> StateGraph:
    """Build and compile the FinSage AI agent graph."""

    graph = StateGraph(FinSageState)

    # Add all nodes
    graph.add_node("supervisor", supervisor_agent.run)
    graph.add_node("stage_1", run_stage_1)
    graph.add_node("stage_2", run_stage_2)
    graph.add_node("stage_3", run_stage_3)
    graph.add_node("review", review_agent.run)
    graph.add_node("synthesis", synthesis_agent.run)

    # Linear pipeline: supervisor → stage_1 → stage_2 → stage_3 → review → synthesis → END
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "stage_1")
    graph.add_edge("stage_1", "stage_2")
    graph.add_edge("stage_2", "stage_3")
    graph.add_edge("stage_3", "review")
    graph.add_edge("review", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


# Compiled graph — import this from API routes and test scripts
app_graph = build_graph()

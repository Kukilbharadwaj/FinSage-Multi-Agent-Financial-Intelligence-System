# agents/graph.py
# LangGraph StateGraph definition — the core orchestration pipeline.
#
# Architecture (v4 — low latency):
#
#   guardrail_in ─┬─ allow ──→ supervisor → stage_1 ⇉ stage_2 ⇉ stage_3
#                 │                              → review → synthesis → guardrail_out → END
#                 └─ smalltalk/block ──→ direct_reply → END
#
# Latency changes vs v3:
#   - Guardrails run through NVIDIA NeMo Guardrails (guardrails/rails.co), but
#     the rails execute deterministic Python actions rather than NeMo's stock
#     self_check_* prompts, so they cost no extra LLM generations.
#   - Small talk ("hi", "what can you do") short-circuits the entire pipeline.
#   - Agents WITHIN a stage now run in parallel threads. They are independent
#     by construction (that is what the stage split encodes), so running them
#     sequentially just added up their latencies for no reason.
#   - The review gate is a local computation instead of an LLM call.
#
# Stage assignment (dependency-aware — order across stages still matters):
#   Stage 1 (independent):     salary, news, general_finance
#   Stage 2 (reads Stage 1):   tax, market
#   Stage 3 (reads Stage 1+2): mutual_fund, trading, technical

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import END, StateGraph

import agents.general_finance_agent as general_finance_agent
import agents.market_agent as market_agent
import agents.mutual_fund_agent as mutual_fund_agent
import agents.news_agent as news_agent
import agents.review_agent as review_agent
import agents.salary_agent as salary_agent
import agents.supervisor_agent as supervisor_agent
import agents.synthesis_agent as synthesis_agent
import agents.tax_agent as tax_agent
import agents.technical_agent as technical_agent
import agents.trading_agent as trading_agent
from agents.nemo_rails import check_input, check_output
from agents.state import FinSageState

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

# The state key each agent owns. Only these keys are merged back after a
# parallel stage, so concurrent agents cannot clobber each other's writes.
AGENT_OUTPUT_KEY = {
    "salary": "salary_analysis",
    "news": "news_analysis",
    "general_finance": "general_finance_result",
    "tax": "tax_analysis",
    "market": "market_analysis",
    "mutual_fund": "mf_analysis",
    "trading": "trading_analysis_output",
    "technical": "technical_analysis",
}

# One agent can take ~10s on a slow upstream API; without a ceiling a single
# hung tool call would stall the whole request.
_AGENT_TIMEOUT_SECONDS = 45


def _run_stage(state: dict, stage_num: int) -> dict:
    """
    Run every selected agent assigned to this stage, in parallel.

    Each agent receives a shallow copy of the state so that concurrent writes
    cannot interleave. Afterwards only the agent's own output key (plus its
    trace lines and any data_freshness stamp) is merged back into the shared
    state, which keeps the communication-bus contract intact.
    """
    selected = state.get("selected_agents", []) or []
    stage_agents = [
        name for name in selected
        if STAGE_ASSIGNMENT.get(name) == stage_num and name in AGENT_REGISTRY
    ]

    if not stage_agents:
        return state

    # A single agent gains nothing from a thread — run it inline.
    if len(stage_agents) == 1:
        name = stage_agents[0]
        try:
            return AGENT_REGISTRY[name](state)
        except Exception as e:
            state["trace"].append(f"stage_{stage_num} → ERROR running {name}: {str(e)[:100]}")
            return state

    def _invoke(agent_name: str) -> tuple:
        branch = copy.copy(state)
        branch["trace"] = []          # collect this agent's trace separately
        branch["rag_context"] = copy.deepcopy(state.get("rag_context"))
        return agent_name, AGENT_REGISTRY[agent_name](branch)

    with ThreadPoolExecutor(max_workers=len(stage_agents)) as pool:
        futures = {pool.submit(_invoke, name): name for name in stage_agents}

        for future in as_completed(futures, timeout=_AGENT_TIMEOUT_SECONDS):
            agent_name = futures[future]
            try:
                name, branch = future.result()
            except Exception as e:
                state["trace"].append(
                    f"stage_{stage_num} → ERROR running {agent_name}: {str(e)[:100]}"
                )
                continue

            # Merge only what this agent owns.
            output_key = AGENT_OUTPUT_KEY.get(name)
            if output_key and branch.get(output_key) is not None:
                state[output_key] = branch[output_key]

            state["trace"].extend(branch.get("trace", []))

            if branch.get("data_freshness"):
                state["data_freshness"] = branch["data_freshness"]

            # RAG context is additive across agents, so fold it in rather than replace.
            branch_rag = branch.get("rag_context")
            if isinstance(branch_rag, dict):
                merged = state.get("rag_context")
                if not isinstance(merged, dict):
                    merged = {"contexts": {}, "sources": [], "combined": ""}
                merged["contexts"].update(branch_rag.get("contexts", {}))
                merged["sources"].extend(branch_rag.get("sources", []))
                merged["combined"] = "\n\n".join(
                    f"[{n}]\n{t}" for n, t in merged["contexts"].items() if t
                )
                state["rag_context"] = merged

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


# ── Guardrail nodes ───────────────────────────────────────────

def guardrail_in(state: dict) -> dict:
    """NeMo input rail: allow, answer small talk directly, or refuse."""
    verdict = check_input(
        state.get("raw_query", ""),
        # Follow-ups are judged against the thread, not in isolation.
        history=state.get("conversation_history") or [],
    )
    action = verdict["action"]

    state["guardrail_action"] = action
    state["input_safe"] = action == "allow"

    if action != "allow":
        state["input_reject_reason"] = verdict["reply"]

    state["trace"].append(
        f"guardrail_in [{verdict.get('engine', 'nemo')}] → {action} ({verdict['reason']})"
    )
    return state


def direct_reply(state: dict) -> dict:
    """Answer small talk / refusals immediately, skipping the agent pipeline."""
    action = state.get("guardrail_action", "block")
    state["recommendation"] = state.get("input_reject_reason") or "Could you rephrase that?"

    if action == "smalltalk":
        state["intent"] = "conversation"
        state["confidence"] = 100
    else:
        state["intent"] = "blocked"
        state["confidence"] = 0

    state["trace"].append(f"direct_reply → answered without running agents ({action})")
    return state


def guardrail_out(state: dict) -> dict:
    """NeMo output rail: soften certainty language and guarantee a disclaimer."""
    recommendation = state.get("recommendation", "")

    if not recommendation or not recommendation.strip():
        state["output_safe"] = True
        state["trace"].append("guardrail_out → skipped (no recommendation)")
        return state

    cleaned, modified = check_output(recommendation)
    state["recommendation"] = cleaned
    state["output_safe"] = True
    state["trace"].append(f"guardrail_out → {'adjusted' if modified else 'clean'}")
    return state


def route_after_guardrail(state: dict) -> str:
    """Route to the full pipeline, or straight to a direct reply."""
    return "supervisor" if state.get("input_safe") else "direct_reply"


def build_graph():
    """Build and compile the FinSage AI agent graph."""

    graph = StateGraph(FinSageState)

    graph.add_node("guardrail_in", guardrail_in)
    graph.add_node("supervisor", supervisor_agent.run)
    graph.add_node("stage_1", run_stage_1)
    graph.add_node("stage_2", run_stage_2)
    graph.add_node("stage_3", run_stage_3)
    graph.add_node("review", review_agent.run)
    graph.add_node("synthesis", synthesis_agent.run)
    graph.add_node("guardrail_out", guardrail_out)
    graph.add_node("direct_reply", direct_reply)

    graph.set_entry_point("guardrail_in")

    graph.add_conditional_edges(
        "guardrail_in",
        route_after_guardrail,
        {
            "supervisor": "supervisor",
            "direct_reply": "direct_reply",
        },
    )

    graph.add_edge("supervisor", "stage_1")
    graph.add_edge("stage_1", "stage_2")
    graph.add_edge("stage_2", "stage_3")
    graph.add_edge("stage_3", "review")
    graph.add_edge("review", "synthesis")
    graph.add_edge("synthesis", "guardrail_out")
    graph.add_edge("guardrail_out", END)

    graph.add_edge("direct_reply", END)

    return graph.compile()


# Compiled graph — import this from API routes and test scripts
app_graph = build_graph()

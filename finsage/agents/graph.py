# agents/graph.py
# LangGraph StateGraph definition and routing logic.
# This is the core orchestration: intent → route → agents → synthesis → END

from langgraph.graph import StateGraph, END
from agents.state import AgentState
import agents.intent_agent as intent_agent
import agents.market_agent as market_agent
import agents.news_agent as news_agent
import agents.technical_agent as technical_agent
import agents.tax_agent as tax_agent
import agents.salary_agent as salary_agent
import agents.trading_agent as trading_agent
import agents.mutual_fund_agent as mutual_fund_agent
import agents.general_finance_agent as general_finance_agent
import agents.synthesis_agent as synthesis_agent


def route_by_intent(state: AgentState) -> str:
    """
    Routing function for conditional edges after intent classification.

    Returns the next node name based on detected intent.
    """
    intent = state.get("intent", "general")

    if intent in ("stock", "index"):
        return "market"
    elif intent == "tax":
        return "tax"
    elif intent == "salary":
        return "salary"
    elif intent == "trading":
        return "trading"
    elif intent == "mutual_fund":
        return "mutual_fund"
    elif intent in ("insurance", "loan", "retirement", "gold", "crypto"):
        return "general_finance"
    else:
        # "general" or any unknown intent
        return "news"


def build_graph() -> StateGraph:
    """Build and compile the FinSage AI agent graph."""

    graph = StateGraph(AgentState)

    # Add all agent nodes
    graph.add_node("intent", intent_agent.run)
    graph.add_node("market", market_agent.run)
    graph.add_node("news", news_agent.run)
    graph.add_node("technical", technical_agent.run)
    graph.add_node("tax", tax_agent.run)
    graph.add_node("salary", salary_agent.run)
    graph.add_node("trading", trading_agent.run)
    graph.add_node("mutual_fund", mutual_fund_agent.run)
    graph.add_node("general_finance", general_finance_agent.run)
    graph.add_node("synthesis", synthesis_agent.run)

    # Entry point
    graph.set_entry_point("intent")

    # Conditional routing after intent classification
    graph.add_conditional_edges(
        "intent",
        route_by_intent,
        {
            "market": "market",
            "news": "news",
            "tax": "tax",
            "salary": "salary",
            "trading": "trading",
            "mutual_fund": "mutual_fund",
            "general_finance": "general_finance",
        },
    )

    # Sequential edges after routing

    # Stock/Index path: market → news → technical → synthesis
    graph.add_edge("market", "news")
    graph.add_edge("news", "technical")
    graph.add_edge("technical", "synthesis")

    # Tax path: tax → synthesis
    graph.add_edge("tax", "synthesis")

    # Salary path: salary → synthesis
    graph.add_edge("salary", "synthesis")

    # Trading path: trading → news → synthesis (news for sentiment context)
    graph.add_edge("trading", "news")

    # News → synthesis (for general and trading paths)
    # Note: news already connects to technical for stock path, 
    # so we need to handle the general/trading case
    # The news node connects differently based on path:
    # - Stock path: news → technical → synthesis
    # - Trading path: news feeds into synthesis via technical
    # Since LangGraph handles this via the path, news → technical handles both

    # Mutual fund path: mutual_fund → synthesis
    graph.add_edge("mutual_fund", "synthesis")

    # General finance path: general_finance → synthesis
    graph.add_edge("general_finance", "synthesis")

    # Synthesis → END
    graph.add_edge("synthesis", END)

    return graph.compile()


# Compiled graph — import this from API routes and test scripts
app_graph = build_graph()

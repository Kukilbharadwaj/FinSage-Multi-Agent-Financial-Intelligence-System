# agents/rag_agent.py
# On-demand RAG knowledge service.
#
# NOT a graph node — this is a utility module that agents call
# when they need knowledge context from the FAISS vector store.
#
# Usage:
#   from agents.rag_agent import retrieve_for_agent
#   rag_text = retrieve_for_agent(state, "tax")
#
# The RAG Agent provides:
#   - Query expansion: builds a focused retrieval query per agent domain
#   - Retrieval: calls query_kb() from rag/knowledge_base.py
#   - Source tracking: appends retrieval metadata to state["rag_context"]

from rag.knowledge_base import query_kb


# Domain-specific query expansion templates.
# Each template appends domain keywords to the user's raw query
# for more focused FAISS retrieval.
_DOMAIN_QUERIES = {
    "tax": "Indian tax 80C 80D STCG LTCG capital gains deduction ITR",
    "salary": "salary budget emergency fund SIP India PPF ELSS NPS savings",
    "mutual_fund": "mutual fund SIP NAV direct plan expense ratio ELSS index fund",
    "market": "Indian stock market analysis valuation sector",
    "trading": "intraday trading options F&O strategies risk management stop-loss",
    "general_finance": "financial planning investment India insurance loan retirement",
}


from observability import observe


@observe()
def retrieve_for_agent(state: dict, agent_name: str, extra_query: str = "") -> str:
    """
    On-demand RAG retrieval for a specific agent.

    Builds a domain-focused query by combining the user's raw query with
    domain keywords, retrieves matching chunks from FAISS, and tracks
    the retrieval in state["rag_context"].

    Args:
        state: The shared FinSageState dict (will be mutated to add context).
        agent_name: Name of the calling agent (e.g. "tax", "salary").
        extra_query: Optional additional keywords to append to the query.

    Returns:
        The retrieved text chunks as a single string (for direct use in prompts).
    """
    raw_query = state.get("raw_query", "")
    domain_keywords = _DOMAIN_QUERIES.get(agent_name, "")

    # Build expanded query: user question + domain keywords + optional extra
    parts = [raw_query, domain_keywords]
    if extra_query:
        parts.append(extra_query)
    expanded_query = " ".join(parts).strip()

    # Retrieve from FAISS
    retrieved_text = query_kb(expanded_query, top_k=4)

    # Initialize rag_context in state if not present
    if not state.get("rag_context") or not isinstance(state["rag_context"], dict):
        state["rag_context"] = {
            "contexts": {},
            "sources": [],
            "combined": "",
        }

    # Store per-agent context
    state["rag_context"]["contexts"][agent_name] = retrieved_text

    # Track the source query for auditing by Review Agent
    state["rag_context"]["sources"].append(
        f"rag_agent({agent_name}): query_kb('{expanded_query[:80]}...')"
    )

    # Update combined context
    all_contexts = state["rag_context"]["contexts"]
    state["rag_context"]["combined"] = "\n\n".join(
        f"[{name}]\n{text}" for name, text in all_contexts.items() if text
    )

    return retrieved_text

# scripts/test_query.py
# Run representative queries straight through the LangGraph graph.
# No server needed.
#
# Usage:
#   python scripts/test_query.py                    # run the standard set
#   python scripts/test_query.py "your question"    # run one ad-hoc query

import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to cp1252 and raise on the rupee sign / arrows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agents.graph import app_graph
from mcp_bridge import startup_mcp_runtime
from observability import get_callbacks, init_langfuse
from rag.knowledge_base import warmup as warm_rag


def build_state(query_text: str) -> dict:
    """Build a clean FinSageState. Keys must match agents/state.py."""
    return {
        # Identity
        "user_id": "test_user",
        "raw_query": query_text,
        # Supervisor outputs
        "goal": "",
        "intent": "",
        "entities": {},
        "selected_agents": [],
        "execution_plan": [],
        # Communication bus
        "salary_analysis": None,
        "news_analysis": None,
        "general_finance_result": None,
        "tax_analysis": None,
        "market_analysis": None,
        "mf_analysis": None,
        "trading_analysis_output": None,
        "technical_analysis": None,
        # RAG + review
        "rag_context": None,
        "review_output": None,
        # Guardrail gate
        "input_safe": None,
        "guardrail_action": None,
        "input_reject_reason": None,
        "output_safe": None,
        # Final output
        "recommendation": None,
        "confidence": None,
        "data_freshness": datetime.now(timezone.utc).isoformat(),
        "trace": [],
    }


def run_query(query_text: str) -> tuple:
    """Run one query through the graph, returning (result, elapsed_seconds)."""
    started = time.time()
    result = app_graph.invoke(
        build_state(query_text),
        config={"callbacks": get_callbacks()},
    )
    return result, time.time() - started


DEFAULT_QUERIES = [
    ("SMALL TALK", "hi, what can you do?"),
    ("BLOCKED", "write me a poem about the sea"),
    ("SALARY", "My monthly salary is 20,000. How should I manage and save?"),
    ("STOCK", "Should I buy Reliance Industries stock right now?"),
    ("INDEX", "What is Nifty 50 at today, and should I buy or wait?"),
    ("TAX", "I sold TCS after 8 months with 50,000 profit. What is my tax?"),
    ("TRADING", "NIFTY option chain - any trade setup for this expiry?"),
    ("MUTUAL FUND", "Best ELSS fund for 80C tax saving with SIP"),
]


def main() -> int:
    # Warm the shared services so the first query is not penalised.
    startup_mcp_runtime()
    warm_rag()
    init_langfuse()

    queries = (
        [("CUSTOM", " ".join(sys.argv[1:]))] if len(sys.argv) > 1 else DEFAULT_QUERIES
    )

    print("=" * 74)
    print("  FinSage AI - Full Agent Graph Test")
    print("=" * 74)

    failures = 0

    for label, query in queries:
        print(f"\n{'-' * 74}")
        print(f"  TEST:  {label}")
        print(f"  QUERY: {query}")
        print(f"{'-' * 74}")

        try:
            result, elapsed = run_query(query)

            print(f"  {elapsed:.2f}s | intent={result.get('intent', 'N/A')} "
                  f"| confidence={result.get('confidence', 'N/A')} "
                  f"| agents={result.get('selected_agents', [])}")

            for step in result.get("trace", []):
                print(f"    - {step[:100]}")

            recommendation = result.get("recommendation") or "No recommendation"
            print()
            for line in recommendation[:600].splitlines():
                print(f"    {line}")
            if len(recommendation) > 600:
                print("    ...")

            if not result.get("recommendation"):
                failures += 1
                print("  [FAIL] no recommendation produced")

        except Exception as e:
            failures += 1
            print(f"  [ERROR] {str(e)[:300]}")

    print("\n" + "=" * 74)
    print("  All tests complete" if failures == 0 else f"  {failures} test(s) had problems")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

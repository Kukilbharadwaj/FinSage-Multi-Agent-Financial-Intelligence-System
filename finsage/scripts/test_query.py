# scripts/test_query.py
# Test all 5 intent paths directly through the LangGraph graph.
# Run this WITHOUT starting the server.
# Usage: python scripts/test_query.py

import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import app_graph


def run_query(query_text: str) -> dict:
    """Build a clean initial state and run the full agent graph."""
    initial_state = {
        "user_id": "test_user",
        "raw_query": query_text,
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
    return app_graph.invoke(initial_state)


def main():
    """Test all 5 query paths."""
    test_queries = [
        ("SALARY PATH", "My monthly salary is ₹20,000. How should I manage and save?"),
        ("STOCK PATH", "Should I buy Reliance Industries stock right now?"),
        ("INDEX PATH", "Nifty 50 is at 22,400. Should I buy or wait?"),
        ("TAX PATH", "I sold TCS after 8 months with ₹50,000 profit. What is my tax?"),
        ("GENERAL PATH", "What is the current Indian market situation?"),
    ]

    print("=" * 70)
    print("  FinSage AI — Full Agent Graph Test")
    print("=" * 70)

    for label, query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"  TEST: {label}")
        print(f"  QUERY: {query}")
        print(f"{'─' * 70}")

        try:
            result = run_query(query)

            print(f"  Intent: {result.get('intent', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}%")
            print(f"  Trace: {' → '.join(result.get('trace', []))}")
            print()

            recommendation = result.get("recommendation", "No recommendation")
            # Show first 500 characters
            if len(recommendation) > 500:
                print(f"  Recommendation (first 500 chars):\n  {recommendation[:500]}...")
            else:
                print(f"  Recommendation:\n  {recommendation}")

        except Exception as e:
            print(f"  ❌ ERROR: {str(e)[:300]}")

        print()

    print("=" * 70)
    print("  All tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

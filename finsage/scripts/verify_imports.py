"""Verify all new/modified modules import correctly."""
import sys
import os
# Ensure finsage dir is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Python: {sys.version}")

# Phase 1: Foundation
try:
    from agents.state import FinSageState
    print("[OK] agents.state.FinSageState")
except Exception as e:
    print(f"[FAIL] agents.state: {e}")

try:
    from agents.rag_agent import retrieve_for_agent
    print("[OK] agents.rag_agent.retrieve_for_agent")
except Exception as e:
    print(f"[FAIL] agents.rag_agent: {e}")

try:
    from agents.supervisor_agent import run as sup_run
    print("[OK] agents.supervisor_agent.run")
except Exception as e:
    print(f"[FAIL] agents.supervisor_agent: {e}")

try:
    from agents.review_agent import run as rev_run
    print("[OK] agents.review_agent.run")
except Exception as e:
    print(f"[FAIL] agents.review_agent: {e}")

# Phase 2-4: All agents
for mod_name in [
    "agents.salary_agent",
    "agents.news_agent",
    "agents.general_finance_agent",
    "agents.tax_agent",
    "agents.market_agent",
    "agents.mutual_fund_agent",
    "agents.trading_agent",
    "agents.technical_agent",
    "agents.synthesis_agent",
]:
    try:
        mod = __import__(mod_name, fromlist=["run"])
        assert hasattr(mod, "run"), "missing run()"
        print(f"[OK] {mod_name}.run")
    except Exception as e:
        print(f"[FAIL] {mod_name}: {e}")

# Phase 5: Graph
try:
    from agents.graph import app_graph
    print(f"[OK] agents.graph.app_graph compiled")
except Exception as e:
    print(f"[FAIL] agents.graph: {e}")

# API routes
try:
    from api.routes import router
    print("[OK] api.routes.router")
except Exception as e:
    print(f"[FAIL] api.routes: {e}")

print("\nDone.")

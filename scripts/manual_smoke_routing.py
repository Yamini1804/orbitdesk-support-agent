"""
Manual smoke check against the REAL local model - not a pytest test.

Loads the actual Retriever + LocalLLM and prints how triage routes a couple
of sample questions, for eyeballing during development.

Deliberately kept out of tests/: Q-003 here is deterministic (it hits the
evidence-marker rule in triage.py and never reaches the LLM), but Q-001 has
no matching rule and falls through to real LLM classification - its route
depends on what Qwen2.5-0.5B-Instruct actually outputs that run, which is
exactly the kind of model-wording dependency the assignment's automated
routing test is required to avoid. See tests/test_graph_routing.py for the
actual assertion-based, model-independent routing tests.

Run directly: python -m scripts.manual_smoke_check
"""
from __future__ import annotations

from src.models import get_llm
from src.nodes.triage import triage_node
from src.retrieval import Retriever
from src.state import new_state

QUERIES = [
    ("Q-001", "Our daily dashboard exports stopped appearing at the expected time "
               "after an Admin changed the workspace timezone yesterday."),
    ("Q-003", "Our data sync is not working. Can you tell me how to fix it?"),
]


def main() -> None:
    print("Loading retriever and local LLM...")
    retriever = Retriever()
    llm = get_llm()

    for qid, question in QUERIES:
        state = new_state(qid, question)
        state["retrieved"] = retriever.search(question)
        result = triage_node(state, llm)
        depends_on_llm = "no rule fired, real LLM classification" if "LLM classification" in result["triage_reason"] else "deterministic rule"
        print(f"{qid} -> {result['route']}  [{depends_on_llm}]")
        print(f"   reason: {result['triage_reason'][:120]}")


if __name__ == "__main__":
    main()
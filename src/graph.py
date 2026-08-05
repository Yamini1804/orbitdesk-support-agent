"""
Graph assembly for the OrbitDesk support agent.

Flow:
    retrieve -> triage -> [conditional: route] -> generate -> verify
                                                                  |
                                          [conditional: passed / retry / failed]
                                                    |         |          |
                                                  END    generate    END

- retrieve runs unconditionally first (cheap, gives triage real evidence).
- triage's `route` field drives which generate path runs (templated vs LLM) -
  see generate.py - but ALL routes still pass through verify, so even the
  templated safety responses get schema-checked, not just trusted blindly.
- verify's pass/fail drives the retry loop. The loop guard is state['retry_count']
  vs MAX_RETRIES (src/state.py), checked inside verify_node itself, so the
  conditional edge here only ever sees two outcomes: "done" or "go back to
  generate" - it can never spin indefinitely because verify_node stops
  emitting "retry" once the ceiling is hit.

Run this file directly (`python -m src.graph`) for a REPL-style smoke test
across the five sample questions.
"""
from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from src.models import LocalLLM, get_llm
from src.nodes.generate import generate_node
from src.nodes.retrieve import retrieve_node
from src.nodes.triage import triage_node
from src.nodes.verify import verify_node
from src.retrieval import Retriever
from src.state import GraphState, new_state


def route_after_verify(state: GraphState) -> str:
    """Pure routing decision - kept at module level (not nested inside
    build_graph) specifically so it's independently importable and unit
    testable without needing a compiled graph, a retriever, or an LLM. See
    tests/test_graph_routing.py.

    verify_node itself enforces the MAX_RETRIES ceiling (see verify.py) - by
    the time we get here, "retry" is only ever returned when it's actually
    safe to loop again. This function just reads that decision."""
    if state.get("verification_passed"):
        return "done"
    if state.get("final_response") is not None:
        # verify_node already produced a safe_failure response - stop.
        return "done"
    return "retry"


def build_graph(retriever: Retriever, llm: LocalLLM):
    """Builds and compiles the StateGraph. Retriever and llm are injected
    (not constructed inside nodes) so they're loaded exactly once, outside
    the graph, and shared across every run - see run.py / __main__ below."""

    graph = StateGraph(GraphState)

    # Each node function needs retriever/llm, but LangGraph node functions
    # only receive `state`. Wrap with closures to bind the shared instances
    # without reloading them per-call or per-question.
    graph.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    graph.add_node("triage", lambda state: triage_node(state, llm))
    graph.add_node("generate", lambda state: generate_node(state, llm))
    graph.add_node("verify", lambda state: verify_node(state))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "triage")
    graph.add_edge("triage", "generate")
    graph.add_edge("generate", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"done": END, "retry": "generate"},
    )

    return graph.compile()


def run_question(compiled_graph, question_id: str, question: str) -> GraphState:
    state = new_state(question_id, question)
    t0 = time.time()
    final_state = compiled_graph.invoke(state)
    elapsed = time.time() - t0
    final_state["_latency_seconds"] = round(elapsed, 2)  # debug-only field, not part of GraphState schema
    return final_state


if __name__ == "__main__":
    import json

    print("Loading retriever and local LLM (one-time cost)...")
    retriever = Retriever()
    llm = get_llm()
    compiled = build_graph(retriever, llm)
    print("Graph compiled.\n")

    sample_questions = json.load(open("data/sample_questions.json"))["questions"]

    for q in sample_questions:
        print(f"{'=' * 70}\n{q['question_id']}: {q['question']}\n{'=' * 70}")
        result = run_question(compiled, q["question_id"], q["question"])

        print(f"[latency] {result['_latency_seconds']}s")
        print("[node trace]")
        for entry in result["node_trace"]:
            print(f"  -> {entry}")

        final = result.get("final_response")
        if final:
            print("\n[final response]")
            print(final.model_dump_json(indent=2))
        else:
            print("\n[WARNING] no final_response produced")
        print()
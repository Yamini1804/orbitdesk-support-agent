"""
Retrieve node - runs BEFORE triage.

Retrieval is cheap (local embeddings, well under a second for this corpus)
and gives triage real evidence to classify against, instead of triage
guessing from the raw question text alone. A question that *sounds* answerable
but retrieves nothing relevant is a strong out_of_scope/clarification signal;
a question that retrieves two overlapping high-confidence KB sections is a
strong answerable signal. See graph.py for how this feeds the triage prompt.
"""
from __future__ import annotations

from src.retrieval import Retriever
from src.state import GraphState, log_node

TOP_K = 4
MIN_SCORE = 0.25


def retrieve_node(state: GraphState, retriever: Retriever) -> dict:
    hits = retriever.search(state["question"], top_k=TOP_K, min_score=MIN_SCORE)

    trace_summary = (
        f"retrieve:{len(hits)} passages"
        + (f" (top={hits[0]['source_id']}@{hits[0]['score']})" if hits else " (no matches above threshold)")
    )

    return {
        "retrieved": hits,
        "node_trace": log_node(state, trace_summary),
    }

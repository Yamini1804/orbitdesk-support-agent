"""
Shared state for the OrbitDesk support graph.

Every node reads a subset of this and returns a partial update (LangGraph
merges dict returns into state - it does NOT require returning the full
state each time). Keeping this as one typed contract is what lets nodes stay
decoupled: triage doesn't need to know how retrieve.py works, it just needs
to know what fields retrieve.py will fill in.

Design choices worth noting in the README:
- `retry_count` + MAX_RETRIES is the infinite-loop guard the assignment
  explicitly requires. It lives in state, not as a global, so it's visible
  in logs and testable.
- `route` is set once by triage and read by the conditional edge function in
  graph.py - routing decisions live in state, not buried in a node's return
  value that only the graph orchestrator sees.
- `node_trace` is the human-readable execution log used in verify.py output,
  the video walkthrough, and the routing test.
"""
from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from src.schema import SupportResponse

MAX_RETRIES = 1  # one revision attempt permitted, per assignment spec

Route = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
]


class RetrievedPassage(TypedDict):
    source_id: str          # KB filename or resolved-case ID
    passage: str             # excerpt text
    score: float              # similarity score, for debugging/logging
    source_type: Literal["knowledge_base", "resolved_case"]
    superseded: bool          # True only for resolved_cases flagged superseded


class GraphState(TypedDict, total=False):
    # --- input ---
    question_id: str
    question: str

    # --- triage output ---
    route: Route
    triage_reason: str
    injection_flagged: bool   # rule-based flag for Q-005-style prompt injection attempts

    # --- retrieval output ---
    retrieved: List[RetrievedPassage]

    # --- generation output ---
    draft_response: Optional[SupportResponse]

    # --- verification output ---
    verification_passed: bool
    verification_notes: List[str]
    retry_count: int

    # --- final ---
    final_response: Optional[SupportResponse]

    # --- observability ---
    node_trace: List[str]      # e.g. ["triage:requires_escalation", "retrieve:2 passages", ...]


def new_state(question_id: str, question: str) -> GraphState:
    """Factory for a fresh state dict at the start of a run."""
    return GraphState(
        question_id=question_id,
        question=question,
        retrieved=[],
        retry_count=0,
        verification_notes=[],
        node_trace=[],
    )


def log_node(state: GraphState, entry: str) -> List[str]:
    """Helper nodes call to append to node_trace without clobbering prior entries.
    Returns the full updated list, ready to assign back into state['node_trace']."""
    return state.get("node_trace", []) + [entry]

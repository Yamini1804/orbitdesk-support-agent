"""
Verify node - runs after generate.

Checks (in order, cheapest/most-certain first):
  1. Schema validity        - draft_response is a valid SupportResponse (should
                               already be true since generate.py constructs one,
                               but re-checked here so a construction bug surfaces
                               as a verification failure with a clear reason,
                               not a silent crash elsewhere in the graph).
  2. Source grounding        - every source_id cited in `sources` was actually
                               retrieved for this question. Catches the model
                               (or a bug) attributing content to a document that
                               was never in evidence.
  3. Has-citation check      - the answer text must contain at least one
                               [SOURCE_ID] tag when evidence was available.
                               Caught in production: the model can pass
                               grounding trivially (sources[] is auto-filled
                               from retrieval) while never actually citing
                               anything in the answer text itself - this check
                               closes that gap.
  4. In-text citation check  - any [KB-xxx]/[CASE-xxx] tag inside the answer
                               body must also correspond to a retrieved
                               source_id - catches fabricated citations.
  5. Unsupported-action leak - the answer must not tell the user OrbitDesk (or
                               the assistant) will perform an action KB-010
                               explicitly rules out (refunds, role changes,
                               contacting external parties, issuing credentials).

On first failure: retry_count is incremented and the graph re-runs generate
once, with the specific failure reason fed back into the prompt (see
generate.py's _retry_feedback) so the retry is a targeted fix, not a blind
repeat. On a second consecutive failure, or if retry_count is already at
MAX_RETRIES, this node returns a safe_failure response instead of asking
graph.py to loop again - the loop-guard is enforced in two places (state
counter + this explicit ceiling) rather than relying on a single point of
failure.
"""
from __future__ import annotations

import re

from src.schema import Classification, Source, SupportResponse
from src.state import MAX_RETRIES, GraphState, log_node

CITATION_TAG_RE = re.compile(r"\[([A-Z]+-\d+)\]")

UNSUPPORTED_ACTION_LEAK_RE = re.compile(
    r"\b(will|have)\s+(issue|process)\s+(a\s+)?refund\b"
    r"|\bi\s+(will|can)\s+(change|update)\s+your\s+(role|account)\b"
    r"|\bi\s+(will|have)\s+contact(ed)?\s+(the\s+)?(external|third[- ]party)\b"
    r"|\bhere\s+is\s+your\s+(api\s+)?(secret|token|password)\b",
    re.IGNORECASE,
)

SAFE_FAILURE_ANSWER = (
    "I wasn't able to produce a response that's fully grounded in the supplied "
    "documentation for this question. Please rephrase with more specific detail, "
    "or this will need review by a human agent."
)


def _check_schema_valid(draft) -> str | None:
    if not isinstance(draft, SupportResponse):
        return "draft_response is not a valid SupportResponse instance."
    return None


def _check_source_grounding(draft: SupportResponse, retrieved: list) -> str | None:
    retrieved_ids = {h["source_id"] for h in retrieved}
    cited_ids = {s.source_id for s in draft.sources}
    ungrounded = cited_ids - retrieved_ids
    if ungrounded:
        return f"Response cites source(s) not present in retrieved evidence: {sorted(ungrounded)}"
    return None


def _check_intext_citations(draft: SupportResponse, retrieved: list) -> str | None:
    retrieved_ids = {h["source_id"] for h in retrieved}
    tags_in_answer = set(CITATION_TAG_RE.findall(draft.answer))
    ungrounded = tags_in_answer - retrieved_ids
    if ungrounded:
        return f"Answer text cites tag(s) not in retrieved evidence: {sorted(ungrounded)}"
    return None


def _check_has_citation(draft: SupportResponse, retrieved: list) -> str | None:
    """The prompt explicitly instructs the model to cite [SOURCE_ID] for every
    fact used. Zero citations in the answer text means we can't actually
    confirm the answer is grounded, even if the `sources` list happens to be
    populated (that list is filled from retrieval, not from what the model
    actually cited - see generate.py). Only applies when evidence existed to
    cite in the first place."""
    if retrieved and not CITATION_TAG_RE.search(draft.answer):
        return "Answer contains no [SOURCE_ID] citation despite retrieved evidence being available."
    return None


def _check_unsupported_action_leak(draft: SupportResponse) -> str | None:
    if UNSUPPORTED_ACTION_LEAK_RE.search(draft.answer):
        return "Answer appears to promise an action the system cannot perform (KB-010)."
    return None


def _run_checks(draft: SupportResponse, retrieved: list, route: str) -> list[str]:
    failures = []

    schema_fail = _check_schema_valid(draft)
    if schema_fail:
        failures.append(schema_fail)
        return failures  # can't run further checks on an invalid object

    # Grounding/citation checks only apply to LLM-generated paths - templated
    # responses (clarification/out_of_scope) are deterministic by construction
    # and don't need re-verifying against retrieval.
    if route in ("answerable", "requires_escalation"):
        for fail in (
            _check_source_grounding(draft, retrieved),
            _check_has_citation(draft, retrieved),
            _check_intext_citations(draft, retrieved),
            _check_unsupported_action_leak(draft),
        ):
            if fail:
                failures.append(fail)

    return failures


def _safe_failure_response(reason: str) -> SupportResponse:
    return SupportResponse(
        classification=Classification.SAFE_FAILURE,
        answer=SAFE_FAILURE_ANSWER,
        sources=[],
        confidence=0.0,
        requires_human=True,
        reason=reason,
        clarification_question=None,
        warnings=["Verification failed after retry limit reached."],
    )


def verify_node(state: GraphState) -> dict:
    draft = state.get("draft_response")
    retrieved = state.get("retrieved", [])
    route = state["route"]
    retry_count = state.get("retry_count", 0)

    failures = _run_checks(draft, retrieved, route)

    if not failures:
        return {
            "verification_passed": True,
            "verification_notes": [],
            "final_response": draft,
            "node_trace": log_node(state, "verify:passed"),
        }

    if retry_count >= MAX_RETRIES:
        safe_response = _safe_failure_response(
            f"Verification failed after {retry_count} retr{'y' if retry_count == 1 else 'ies'}: "
            + "; ".join(failures)
        )
        return {
            "verification_passed": False,
            "verification_notes": failures,
            "final_response": safe_response,
            "node_trace": log_node(state, f"verify:failed->safe_failure ({'; '.join(failures)})"),
        }

    # Under retry limit - signal graph.py to route back to generate once more.
    return {
        "verification_passed": False,
        "verification_notes": failures,
        "retry_count": retry_count + 1,
        "final_response": None,
        "node_trace": log_node(state, f"verify:failed->retry ({'; '.join(failures)})"),
    }
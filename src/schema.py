"""
Structured response contract for the OrbitDesk support agent.

This mirrors data/output_schema.json field-for-field. The verification node
validates against this model rather than the raw dict, so a malformed
response is a caught ValidationError, not a silent bad output.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Classification(str, Enum):
    ANSWERABLE = "answerable"
    REQUIRES_CLARIFICATION = "requires_clarification"
    REQUIRES_ESCALATION = "requires_escalation"
    OUT_OF_SCOPE = "out_of_scope"
    SAFE_FAILURE = "safe_failure"


class Source(BaseModel):
    source_id: str = Field(..., description="KB document ID or resolved-case ID")
    passage: str = Field(..., min_length=1, description="Relevant excerpt or passage identifier")


class SupportResponse(BaseModel):
    classification: Classification
    answer: str = Field(..., min_length=1)
    sources: List[Source] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_human: bool
    reason: str = Field(..., min_length=1, description="Brief explanation of the route and confidence")
    clarification_question: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def answerable_requires_sources(self):
        """An 'answerable' response with zero sources is a grounding failure,
        not a valid response - catch it here so verify.py can rely on schema
        validity alone meaning "well-formed AND grounded", not just well-formed."""
        if self.classification == Classification.ANSWERABLE and not self.sources:
            raise ValueError("classification='answerable' requires at least one source")
        return self

    class Config:
        use_enum_values = True

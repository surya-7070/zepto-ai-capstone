"""
Module 3, Task 5 — the enforced JSON output schema.

Every answer leaving the graph, and therefore every response leaving the
FastAPI endpoint, is validated against AnswerResponse.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    query: str = Field(..., min_length=1, description="The customer's question")


class AnswerResponse(BaseModel):
    """The validated answer contract."""

    answer: str = Field(..., description="The reply shown to the customer")
    sources: list[str] = Field(
        default_factory=list,
        description="Chunk/document ids used; empty for general_question answers",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the answer, 0-1"
    )

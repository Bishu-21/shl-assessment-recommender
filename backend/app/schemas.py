"""Pydantic schemas for the SHL Assessment Recommender API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation."""

    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message text")


class ChatRequest(BaseModel):
    """Incoming chat request – full stateless conversation history."""

    messages: list[Message] = Field(
        ...,
        description="Full conversation history, alternating user/assistant turns",
        min_length=1,
    )


class Recommendation(BaseModel):
    """A single assessment recommendation from the SHL catalog."""

    name: str = Field(..., description="Assessment name from the SHL catalog")
    url: str = Field(..., description="Product catalog URL")
    test_type: str = Field(
        ...,
        description="Test type code(s), e.g. 'A', 'K', 'P', 'B', 'S', 'C', 'D'",
    )
    keys: str = Field(
        ...,
        description="Assessment category keys, e.g. 'Knowledge & Skills'",
    )
    duration: str = Field("", description="Approximate completion time")
    languages: str = Field("", description="Available languages")


class ChatResponse(BaseModel):
    """Response from the /chat endpoint."""

    reply: str = Field(..., description="Conversational reply text")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="List of recommended assessments (empty array when not recommending)",
    )
    end_of_conversation: bool = Field(
        False,
        description="Whether the conversation has reached a natural conclusion",
    )


class HealthResponse(BaseModel):
    """Response from /health."""

    status: str = "ok"

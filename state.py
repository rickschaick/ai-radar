"""Shared state and data models for the AI Radar graph."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

Category = Literal["frontier", "productie", "cases", "hype"]
Region = Literal["nl", "int"]


class ScoutItem(BaseModel):
    """A single finding collected by a scout agent."""

    title: str = Field(description="Short title of the finding")
    summary: str = Field(description="One to three sentence summary of why this matters")
    source_url: str = Field(description="Source URL")
    date: str | None = Field(default=None, description="Publication date (YYYY-MM-DD) if known")
    category: Category = Field(description="Which scout found this item")
    region: Region = Field(default="int", description="'nl' = Dutch news, 'int' = international")
    hype_score: int = Field(
        ge=1, le=5,
        description="1 = sober, evidence-backed; 5 = pure hype / unverified claims",
    )
    evidence: str = Field(
        description="What concrete evidence backs the claim (benchmarks, data, "
        "named customers, regulation text) — or 'none' if it is just a claim"
    )


class ScoutReport(BaseModel):
    """Structured output returned by a scout LLM call."""

    items: list[ScoutItem]


class CriticVerdict(BaseModel):
    """Structured output returned by the critic."""

    approved: bool = Field(description="True if the briefing is ready to publish")
    feedback: str = Field(description="Concrete, actionable feedback in Dutch")


class RadarState(TypedDict, total=False):
    """State passed between nodes in the LangGraph StateGraph."""

    # Input
    date: str

    # Scout outputs (separate keys so the scouts can run in parallel)
    frontier_items: list[ScoutItem]
    productie_items: list[ScoutItem]
    case_items: list[ScoutItem]
    hype_items: list[ScoutItem]

    # Synthesizer / critic loop
    draft_briefing: str
    critique: str
    approved: bool
    retries: int

    # Output
    final_briefing: str
    article: str

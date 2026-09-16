from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    task: str = Field(
        default="Find completed clinical trials with research integrity issues",
        description="The analysis task in english. Agents read this and decide what to investigate.",
        examples="Find completed trials where sponsor never posted results",
    )

    nct_ids: list[str] = Field(
        default=[],
        description="Specific NCT IDs to analyse. Empty list means analyse broadly across all studies.",
        examples=["NCT04788680", "NCT02208921"]
    )

    max_studies: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum studies to analyse per agent. Default 10.",
    )


class ReviewDecisionRequest(BaseModel):
    decision: str = Field(
        description="The reviewer's decision: approve, reject or edit",
        examples="reject",
    )

    reviewer: str = Field(
        default="analyst",
        description="Name or ID of the human reviewer.",
        examples="sahil@gmail.com"
    )

    rejection_reason: str = Field(
        default="",
        description="Why this signal was rejected. IMPORTANT: This gets written to procedural memory and permanently changes how the agent reasons. Be specific and clear.",
        examples="This trial was terminated early due to COVID - terminated trials are exempt from result posting requirements.",
    )

    edit_summary: str = Field(
        default="",
        description="Corrected signal summary if decision is 'edit'. Replaces the agent's original summary.",
    )


class SignalResponse(BaseModel):
    signal_id: str
    nct_id: str
    agent: str
    signal_type: str
    summary: str
    confidence: float
    status: str
    created_at: str

    class Config:
        from_attributes = True


class AnalysisResponse(BaseModel):
    run_id: str
    task: str
    final_brief: str
    total_signals: int
    signals_requiring_review: int
    agents_activated: list[str]
    duration_seconds: float


class ReviewQueueItem(BaseModel):
    review_id: str
    signal_id: str
    agent: str
    signal_type: str
    summary: str
    confidence: float
    nct_id: str
    decision: str


class ReviewQueueResponse(BaseModel):
    queue: list[ReviewQueueItem]
    total_pending: int
    total_approved: int
    total_rejected: int


class ReviewDecisionResponse(BaseModel):
    success: bool
    decision: str
    signal_id: str
    queue_id: str
    memory_updated: bool
    message: str


class EpisodeResponse(BaseModel):
    episode_id: str
    agent_name: str
    nct_id: str | None
    content: str
    outcome: str | None
    similarity: float | None
    created_at: str


class ProcedureResponse(BaseModel):
    procedure_id: str
    agent_name: str
    rule_text: str
    rule_type: str
    source: str
    created_at: str


class SponsorProfileResponse(BaseModel):
    sponsor: str
    credibility_score: float
    total_studies: int
    results_posted: int
    results_missing: int
    broken_promises: int
    avg_delay_days: float
    last_updated: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    database: str
    details: dict[str, Any]

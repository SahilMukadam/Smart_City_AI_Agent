"""
Smart City AI Agent - Structured Response Models
Rich response format for the agent API that includes confidence,
source attribution, timing, and correlation insights.
"""

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """Information about a single data source used."""
    tool_name: str
    success: bool
    cached: bool = False
    response_time_ms: float = 0.0
    error: str | None = None


class CorrelationInsight(BaseModel):
    """A single correlation insight for the API response."""
    type: str
    title: str
    description: str
    confidence: str


class AgentResponse(BaseModel):
    """
    Structured response from the agent.
    This is what the frontend receives — rich metadata
    alongside the analysis text.
    """
    # ── Core Response ─────────────────────────────────────────────
    response: str = Field(description="Agent's analysis and answer")
    success: bool = Field(description="Whether the agent completed successfully")

    # ── Session ───────────────────────────────────────────────────
    session_id: str = Field(description="Session ID for follow-up queries")

    # ── Tool Metadata ─────────────────────────────────────────────
    tools_used: list[str] = Field(
        default_factory=list,
        description="Tools the agent selected",
    )
    sources: list[SourceInfo] = Field(
        default_factory=list,
        description="Detailed info about each data source",
    )

    # ── Correlation Insights ──────────────────────────────────────
    insights: list[CorrelationInsight] = Field(
        default_factory=list,
        description="Cross-source correlation insights detected",
    )

    # ── Timing ────────────────────────────────────────────────────
    total_time_ms: float = Field(
        default=0.0,
        description="Total request processing time in milliseconds",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(),
        description="Response timestamp (UTC)",
    )

    # ── Error ─────────────────────────────────────────────────────
    error: str | None = Field(
        default=None,
        description="Error message if something went wrong",
    )

    # ── Cache ─────────────────────────────────────────────────────
    cache_stats: dict | None = Field(
        default=None,
        description="Cache hit/miss info for this request",
    )

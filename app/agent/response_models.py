"""
Smart City AI Agent - Structured Response Models (Day 9)
"""

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    tool_name: str
    success: bool
    cached: bool = False
    response_time_ms: float = 0.0
    error: str | None = None


class CorrelationInsight(BaseModel):
    type: str
    title: str
    description: str
    confidence: str


class AnomalyAlert(BaseModel):
    """A detected anomaly for the API response."""
    level: str  # info, warning, critical
    category: str  # traffic, weather, air_quality, tube
    title: str
    description: str
    metric: str
    current_value: str
    threshold: str
    recommendation: str = ""


class HealthScores(BaseModel):
    """City health scores per category."""
    overall: int | None = None
    traffic: int | None = None
    weather: int | None = None
    air_quality: int | None = None
    tube: int | None = None


class AgentResponse(BaseModel):
    """Structured response from the agent."""
    response: str = Field(description="Agent's analysis and answer")
    success: bool
    session_id: str

    tools_used: list[str] = Field(default_factory=list)
    sources: list[SourceInfo] = Field(default_factory=list)

    insights: list[CorrelationInsight] = Field(default_factory=list)
    anomalies: list[AnomalyAlert] = Field(default_factory=list)
    health: HealthScores | None = None

    total_time_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    error: str | None = None
    cache_stats: dict | None = None

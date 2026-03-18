"""
Smart City AI Agent - Response Schemas
Standardized response format for all data source tools.
Every tool wrapper returns a ToolResponse so the LLM agent
can parse results consistently regardless of data source.
"""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    """
    Standardized response from any data source tool.
    The LLM agent uses `summary` for quick understanding
    and `data` for detailed analysis when needed.
    """
    tool_name: str = Field(
        description="Identifier: 'tfl', 'weather', 'air_quality', 'tomtom'"
    )
    query_type: str = Field(
        description="What was queried: 'tube_status', 'disruptions', 'road_status', etc."
    )
    success: bool = Field(
        description="Whether the API call succeeded"
    )
    data: Any = Field(
        default=None,
        description="Structured response data (parsed and cleaned)"
    )
    summary: str = Field(
        default="",
        description="Human-readable summary for the LLM agent to use directly"
    )
    error: str | None = Field(
        default=None,
        description="Error message if success=False"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="When this data was fetched"
    )
    source_url: str = Field(
        default="",
        description="The API endpoint that was called"
    )
    response_time_ms: float = Field(
        default=0.0,
        description="How long the API call took in milliseconds"
    )

    def to_agent_string(self) -> str:
        """
        Format this response as a string the LLM agent can easily consume.
        Used when passing tool results back into the agent's context.
        """
        if not self.success:
            return f"[{self.tool_name}:{self.query_type}] ERROR: {self.error}"

        parts = [
            f"[{self.tool_name}:{self.query_type}]",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Summary: {self.summary}",
        ]
        return "\n".join(parts)


class TubeLineStatus(BaseModel):
    """Status of a single TfL tube line."""
    name: str
    status: str
    severity: int = Field(description="TfL severity code (10=Good, 0=Closed)")
    reason: str | None = None


class RoadDisruption(BaseModel):
    """A single road disruption event."""
    id: str
    severity: str
    category: str
    location: str
    comments: str = ""
    road_name: str = ""
    start_date: str | None = None
    end_date: str | None = None


class RoadCorridor(BaseModel):
    """Status of a major road corridor."""
    id: str
    name: str
    status: str
    severity: str
    status_description: str = ""

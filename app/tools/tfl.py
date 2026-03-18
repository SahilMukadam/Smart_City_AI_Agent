"""
Smart City AI Agent - Transport for London (TfL) API Tool
Fetches real-time tube status, road disruptions, and road corridor data.

TfL API: https://api.tfl.gov.uk (free, no key required for basic access)
Optional: Set TFL_APP_KEY in .env for higher rate limits.
"""

import logging
from datetime import datetime, timezone

import httpx

from app.config import get_settings
from app.tools.base import BaseTool
from app.models.schemas import (
    ToolResponse,
    TubeLineStatus,
    RoadDisruption,
    RoadCorridor,
)

logger = logging.getLogger(__name__)


class TfLTool(BaseTool):
    """
    Transport for London data tool.
    Provides real-time tube status, road disruptions, and road corridor info.
    """

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self._base_url = settings.TFL_BASE_URL
        self._app_key = settings.TFL_APP_KEY

    @property
    def name(self) -> str:
        return "tfl"

    @property
    def description(self) -> str:
        return (
            "Transport for London (TfL) data tool. "
            "Fetches real-time London tube/underground line status, "
            "road disruptions and incidents, and major road corridor conditions. "
            "Use this when the user asks about London transport, tube delays, "
            "road closures, or traffic disruptions."
        )

    def get_capabilities(self) -> list[str]:
        return ["tube_status", "road_disruptions", "road_status"]

    def _get_params(self) -> dict:
        """Build query params, adding app_key if configured."""
        params = {}
        if self._app_key:
            params["app_key"] = self._app_key
        return params

    # ── Tube Status ───────────────────────────────────────────────

    def get_tube_status(self) -> ToolResponse:
        """
        Fetch current status of all London Underground lines.
        Returns severity levels and disruption reasons.
        """
        url = f"{self._base_url}/Line/Mode/tube/Status"
        query_type = "tube_status"

        try:
            raw_data, elapsed_ms = self._timed_request(url, params=self._get_params())
            lines = self._parse_tube_status(raw_data)

            disrupted = [l for l in lines if l.severity < 10]
            disrupted_count = len(disrupted)
            total_lines = len(lines)

            summary = self._build_tube_summary(lines, disrupted_count, total_lines)

            return ToolResponse(
                tool_name=self.name,
                query_type=query_type,
                success=True,
                data={
                    "lines": [l.model_dump() for l in lines],
                    "disrupted_count": disrupted_count,
                    "total_lines": total_lines,
                },
                summary=summary,
                timestamp=datetime.now(tz=timezone.utc),
                source_url=url,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Failed to fetch tube status: {e}")
            return self._build_error_response(query_type, e, url)

    def _parse_tube_status(self, raw_data: list[dict]) -> list[TubeLineStatus]:
        """Parse TfL tube status response into clean models."""
        lines = []
        for line in raw_data:
            # TfL returns a list of statuses; take the first (most relevant)
            status_info = line.get("lineStatuses", [{}])[0]
            lines.append(
                TubeLineStatus(
                    name=line.get("name", "Unknown"),
                    status=status_info.get("statusSeverityDescription", "Unknown"),
                    severity=status_info.get("statusSeverity", -1),
                    reason=status_info.get("reason"),
                )
            )
        return lines

    def _build_tube_summary(
        self,
        lines: list[TubeLineStatus],
        disrupted_count: int,
        total: int,
    ) -> str:
        """Build a human-readable summary of tube status."""
        if disrupted_count == 0:
            return f"All {total} tube lines have good service."

        parts = [f"{disrupted_count} of {total} tube lines have disruptions:"]
        for line in lines:
            if line.severity < 10:
                reason_str = f" ({line.reason})" if line.reason else ""
                parts.append(f"  - {line.name}: {line.status}{reason_str}")

        return "\n".join(parts)

    # ── Road Disruptions ──────────────────────────────────────────

    def get_road_disruptions(self) -> ToolResponse:
        """
        Fetch all current road disruptions across London.
        Includes planned works, incidents, and closures.
        """
        url = f"{self._base_url}/Road/all/Disruption"
        query_type = "road_disruptions"

        try:
            raw_data, elapsed_ms = self._timed_request(url, params=self._get_params())
            disruptions = self._parse_disruptions(raw_data)

            summary = self._build_disruption_summary(disruptions)

            return ToolResponse(
                tool_name=self.name,
                query_type=query_type,
                success=True,
                data={
                    "disruptions": [d.model_dump() for d in disruptions],
                    "total_count": len(disruptions),
                    "by_severity": self._count_by_severity(disruptions),
                },
                summary=summary,
                timestamp=datetime.now(tz=timezone.utc),
                source_url=url,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Failed to fetch road disruptions: {e}")
            return self._build_error_response(query_type, e, url)

    def _parse_disruptions(self, raw_data: list[dict]) -> list[RoadDisruption]:
        """Parse TfL road disruption response."""
        disruptions = []
        for item in raw_data:
            disruptions.append(
                RoadDisruption(
                    id=item.get("id", ""),
                    severity=item.get("severity", "Unknown"),
                    category=item.get("category", "Unknown"),
                    location=item.get("location", "Unknown location"),
                    comments=item.get("comments", ""),
                    road_name=self._extract_road_name(item),
                    start_date=item.get("startDate"),
                    end_date=item.get("endDate"),
                )
            )
        return disruptions

    def _extract_road_name(self, disruption: dict) -> str:
        """Extract road name from disruption data. TfL nests this differently."""
        # Try direct field first
        if road := disruption.get("corridorIds"):
            if isinstance(road, list) and road:
                return road[0]
        # Fallback: check location string for road reference
        location = disruption.get("location", "")
        if "/" in location:
            return location.split("/")[0].strip()
        return ""

    def _count_by_severity(self, disruptions: list[RoadDisruption]) -> dict[str, int]:
        """Count disruptions by severity level."""
        counts: dict[str, int] = {}
        for d in disruptions:
            counts[d.severity] = counts.get(d.severity, 0) + 1
        return counts

    def _build_disruption_summary(self, disruptions: list[RoadDisruption]) -> str:
        """Build human-readable disruption summary."""
        total = len(disruptions)
        if total == 0:
            return "No road disruptions currently reported in London."

        severity_counts = self._count_by_severity(disruptions)
        severity_str = ", ".join(
            f"{count} {sev}" for sev, count in severity_counts.items()
        )

        parts = [f"{total} road disruptions currently active ({severity_str})."]

        # Show first 5 most relevant disruptions
        for d in disruptions[:5]:
            location_str = d.location[:80] if d.location else "Unknown location"
            parts.append(f"  - [{d.severity}] {d.category}: {location_str}")

        if total > 5:
            parts.append(f"  ... and {total - 5} more.")

        return "\n".join(parts)

    # ── Road Corridor Status ──────────────────────────────────────

    def get_road_status(self, road_ids: str | None = None) -> ToolResponse:
        """
        Fetch status of major road corridors.
        Args:
            road_ids: Comma-separated road IDs (e.g., "A1,A2,A40").
                       If None, fetches all major roads.
        """
        if road_ids:
            url = f"{self._base_url}/Road/{road_ids}/Status"
        else:
            url = f"{self._base_url}/Road"
        query_type = "road_status"

        try:
            raw_data, elapsed_ms = self._timed_request(url, params=self._get_params())
            corridors = self._parse_road_status(raw_data)

            summary = self._build_road_summary(corridors, road_ids)

            return ToolResponse(
                tool_name=self.name,
                query_type=query_type,
                success=True,
                data={
                    "corridors": [c.model_dump() for c in corridors],
                    "total_count": len(corridors),
                },
                summary=summary,
                timestamp=datetime.now(tz=timezone.utc),
                source_url=url,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Failed to fetch road status: {e}")
            return self._build_error_response(query_type, e, url)

    def _parse_road_status(self, raw_data: list[dict]) -> list[RoadCorridor]:
        """Parse TfL road corridor response."""
        corridors = []
        for road in raw_data:
            corridors.append(
                RoadCorridor(
                    id=road.get("id", ""),
                    name=road.get("displayName", road.get("id", "Unknown")),
                    status=road.get("statusSeverity", "Unknown"),
                    severity=road.get("statusSeverity", "Unknown"),
                    status_description=road.get(
                        "statusSeverityDescription", ""
                    ),
                )
            )
        return corridors

    def _build_road_summary(
        self,
        corridors: list[RoadCorridor],
        road_ids: str | None,
    ) -> str:
        """Build human-readable road status summary."""
        total = len(corridors)
        if total == 0:
            return "No road corridor data available."

        label = f"roads ({road_ids})" if road_ids else "major road corridors"
        parts = [f"Status of {total} {label}:"]

        for c in corridors[:10]:
            desc = f" - {c.status_description}" if c.status_description else ""
            parts.append(f"  - {c.name}: {c.status}{desc}")

        if total > 10:
            parts.append(f"  ... and {total - 10} more.")

        return "\n".join(parts)

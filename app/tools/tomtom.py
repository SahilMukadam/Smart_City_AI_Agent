"""
Smart City AI Agent - TomTom Traffic Flow Tool
Fetches real-time traffic speed, congestion, and incidents.

TomTom API: https://developer.tomtom.com (free tier: 2,500 req/day)
Provides: current speed vs free-flow speed, congestion ratios, incidents.
"""

import logging
from datetime import datetime, timezone

from app.config import get_settings
from app.tools.base import BaseTool
from app.models.schemas import ToolResponse

logger = logging.getLogger(__name__)

# ── London Key Locations ──────────────────────────────────────────
# Pre-defined points for common London queries
LONDON_POINTS = {
    "central": {"lat": 51.5074, "lon": -0.1278, "label": "Central London"},
    "city": {"lat": 51.5155, "lon": -0.0922, "label": "City of London"},
    "westminster": {"lat": 51.4975, "lon": -0.1357, "label": "Westminster"},
    "camden": {"lat": 51.5390, "lon": -0.1426, "label": "Camden"},
    "tower_bridge": {"lat": 51.5055, "lon": -0.0754, "label": "Tower Bridge"},
    "kings_cross": {"lat": 51.5317, "lon": -0.1240, "label": "King's Cross"},
    "canary_wharf": {"lat": 51.5054, "lon": -0.0235, "label": "Canary Wharf"},
    "shoreditch": {"lat": 51.5274, "lon": -0.0777, "label": "Shoreditch"},
    "brixton": {"lat": 51.4613, "lon": -0.1156, "label": "Brixton"},
    "hammersmith": {"lat": 51.4927, "lon": -0.2248, "label": "Hammersmith"},
}

# ── London Bounding Box (for incidents) ───────────────────────────
LONDON_BBOX = {
    "min_lat": 51.28,
    "min_lon": -0.51,
    "max_lat": 51.69,
    "max_lon": 0.33,
}

# ── Congestion Level Interpretation ───────────────────────────────
def _classify_congestion(ratio: float) -> str:
    """
    Classify congestion based on current_speed / free_flow_speed ratio.
    ratio = 1.0 means free-flowing, ratio = 0.0 means standstill.
    """
    if ratio >= 0.85:
        return "Free flowing"
    elif ratio >= 0.65:
        return "Light congestion"
    elif ratio >= 0.40:
        return "Moderate congestion"
    elif ratio >= 0.20:
        return "Heavy congestion"
    else:
        return "Severe congestion / Standstill"


class TomTomTool(BaseTool):
    """
    TomTom traffic data tool.
    Provides real-time traffic flow (speed, congestion) and incident data.
    """

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self._base_url = settings.TOMTOM_BASE_URL
        self._api_key = settings.TOMTOM_API_KEY

    @property
    def name(self) -> str:
        return "tomtom"

    @property
    def description(self) -> str:
        return (
            "TomTom traffic data tool. "
            "Fetches real-time traffic flow data including current speed, "
            "free-flow speed, congestion levels, and travel times for specific "
            "road segments. Also fetches traffic incidents (accidents, roadworks, "
            "closures) in an area. "
            "Use this when the user asks about traffic speed, congestion levels, "
            "travel times, or real-time road conditions at specific locations. "
            "Complements TfL data with actual speed measurements."
        )

    def get_capabilities(self) -> list[str]:
        return ["traffic_flow", "traffic_incidents", "multi_point_flow"]

    def _check_api_key(self) -> str | None:
        """Return error message if API key is not configured."""
        if not self._api_key:
            return "TomTom API key not configured. Set TOMTOM_API_KEY in .env"
        return None

    # ── Traffic Flow (single point) ───────────────────────────────

    def get_traffic_flow(
        self,
        latitude: float = 51.5074,
        longitude: float = -0.1278,
        location_name: str | None = None,
    ) -> ToolResponse:
        """
        Fetch real-time traffic flow data for a road segment near a point.
        Returns current speed, free-flow speed, congestion ratio.
        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Optional human-readable name for summaries
        """
        query_type = "traffic_flow"

        # Check API key
        if err := self._check_api_key():
            return self._build_error_response(query_type, Exception(err))

        # TomTom Flow Segment endpoint
        # style=absolute gives actual speeds, zoom=10 is good for city streets
        url = (
            f"{self._base_url}/traffic/services/4/flowSegmentData"
            f"/absolute/10/json"
        )
        params = {
            "key": self._api_key,
            "point": f"{latitude},{longitude}",
            "unit": "KMPH",
            "thickness": 1,
        }

        try:
            raw_data, elapsed_ms = self._timed_request(url, params=params)
            parsed = self._parse_flow_data(raw_data, location_name, latitude, longitude)
            summary = self._build_flow_summary(parsed)

            return ToolResponse(
                tool_name=self.name,
                query_type=query_type,
                success=True,
                data=parsed,
                summary=summary,
                timestamp=datetime.now(tz=timezone.utc),
                source_url=url,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Failed to fetch traffic flow: {e}")
            return self._build_error_response(query_type, e, url)

    def _parse_flow_data(
        self,
        raw_data: dict,
        location_name: str | None,
        latitude: float,
        longitude: float,
    ) -> dict:
        """Parse TomTom flow segment response."""
        flow = raw_data.get("flowSegmentData", {})
        current_speed = flow.get("currentSpeed", 0)
        free_flow_speed = flow.get("freeFlowSpeed", 0)
        current_travel_time = flow.get("currentTravelTime", 0)
        free_flow_travel_time = flow.get("freeFlowTravelTime", 0)
        confidence = flow.get("confidence", 0)
        road_closure = flow.get("roadClosure", False)

        # Calculate congestion ratio
        if free_flow_speed > 0:
            congestion_ratio = round(current_speed / free_flow_speed, 3)
        else:
            congestion_ratio = 1.0

        # Calculate delay
        delay_seconds = max(0, current_travel_time - free_flow_travel_time)

        # Classify congestion level
        congestion_level = "Road closed" if road_closure else _classify_congestion(congestion_ratio)

        label = location_name or self._get_location_label(latitude, longitude)

        return {
            "location": label,
            "latitude": latitude,
            "longitude": longitude,
            "current_speed_kmh": current_speed,
            "free_flow_speed_kmh": free_flow_speed,
            "congestion_ratio": congestion_ratio,
            "congestion_level": congestion_level,
            "current_travel_time_s": current_travel_time,
            "free_flow_travel_time_s": free_flow_travel_time,
            "delay_seconds": delay_seconds,
            "road_closure": road_closure,
            "confidence": confidence,
            "functional_road_class": flow.get("functionalRoadClass", ""),
        }

    def _build_flow_summary(self, parsed: dict) -> str:
        """Build human-readable traffic flow summary."""
        location = parsed["location"]
        level = parsed["congestion_level"]
        current = parsed["current_speed_kmh"]
        free_flow = parsed["free_flow_speed_kmh"]
        ratio = parsed["congestion_ratio"]
        delay = parsed["delay_seconds"]

        if parsed["road_closure"]:
            return f"Traffic near {location}: ROAD CLOSED."

        parts = [
            f"Traffic near {location}: {level}.",
            f"Current speed: {current} km/h (free-flow: {free_flow} km/h).",
            f"Congestion ratio: {ratio:.0%}.",
        ]
        if delay > 0:
            parts.append(f"Additional delay: {delay}s above normal travel time.")

        return " ".join(parts)

    # ── Multi-Point Flow ──────────────────────────────────────────

    def get_multi_point_flow(
        self,
        points: list[str] | None = None,
    ) -> ToolResponse:
        """
        Fetch traffic flow for multiple London locations at once.
        Args:
            points: List of location keys from LONDON_POINTS
                    (e.g., ["central", "city", "canary_wharf"]).
                    If None, queries all predefined points.
        """
        query_type = "multi_point_flow"

        if err := self._check_api_key():
            return self._build_error_response(query_type, Exception(err))

        # Resolve location keys
        if points is None:
            points = list(LONDON_POINTS.keys())

        results = []
        errors = []

        for point_key in points:
            point_info = LONDON_POINTS.get(point_key)
            if not point_info:
                errors.append(f"Unknown location: {point_key}")
                continue

            flow = self.get_traffic_flow(
                latitude=point_info["lat"],
                longitude=point_info["lon"],
                location_name=point_info["label"],
            )

            if flow.success:
                results.append(flow.data)
            else:
                errors.append(f"{point_info['label']}: {flow.error}")

        # Sort by congestion (worst first)
        results.sort(key=lambda x: x.get("congestion_ratio", 1.0))

        summary = self._build_multi_flow_summary(results, errors)

        return ToolResponse(
            tool_name=self.name,
            query_type=query_type,
            success=True,
            data={
                "points": results,
                "total_queried": len(points),
                "successful": len(results),
                "errors": errors,
            },
            summary=summary,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def _build_multi_flow_summary(
        self,
        results: list[dict],
        errors: list[str],
    ) -> str:
        """Build summary for multi-point traffic flow."""
        if not results:
            return "No traffic flow data retrieved."

        parts = [f"Traffic flow at {len(results)} London locations (worst congestion first):"]
        for r in results:
            location = r["location"]
            level = r["congestion_level"]
            speed = r["current_speed_kmh"]
            ratio = r["congestion_ratio"]
            parts.append(f"  - {location}: {level} ({speed} km/h, {ratio:.0%} of free-flow)")

        if errors:
            parts.append(f"\nFailed for {len(errors)} location(s): {'; '.join(errors)}")

        return "\n".join(parts)

    # ── Traffic Incidents ─────────────────────────────────────────

    def get_traffic_incidents(
        self,
        min_lat: float = LONDON_BBOX["min_lat"],
        min_lon: float = LONDON_BBOX["min_lon"],
        max_lat: float = LONDON_BBOX["max_lat"],
        max_lon: float = LONDON_BBOX["max_lon"],
    ) -> ToolResponse:
        """
        Fetch traffic incidents (accidents, roadworks, closures) in a bounding box.
        Defaults to Greater London area.
        """
        query_type = "traffic_incidents"

        if err := self._check_api_key():
            return self._build_error_response(query_type, Exception(err))

        url = (
            f"{self._base_url}/traffic/services/5/incidentDetails"
        )
        params = {
            "key": self._api_key,
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "fields": (
                "{incidents{type,geometry{type,coordinates},"
                "properties{id,iconCategory,magnitudeOfDelay,"
                "startTime,endTime,from,to,length,delay,"
                "roadNumbers,timeValidity,probabilityOfOccurrence,"
                "events{description,code}}}}"
            ),
            "language": "en-GB",
            "categoryFilter": "0,1,2,3,4,5,6,7,8,9,10,11,14",
            "timeValidityFilter": "present",
        }

        try:
            raw_data, elapsed_ms = self._timed_request(url, params=params)
            incidents = self._parse_incidents(raw_data)
            summary = self._build_incidents_summary(incidents)

            return ToolResponse(
                tool_name=self.name,
                query_type=query_type,
                success=True,
                data={
                    "incidents": incidents,
                    "total_count": len(incidents),
                    "by_category": self._count_by_category(incidents),
                },
                summary=summary,
                timestamp=datetime.now(tz=timezone.utc),
                source_url=url,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Failed to fetch traffic incidents: {e}")
            return self._build_error_response(query_type, e, url)

    def _parse_incidents(self, raw_data: dict) -> list[dict]:
        """Parse TomTom incidents response."""
        incidents_raw = raw_data.get("incidents", [])
        incidents = []

        # TomTom icon categories
        category_map = {
            0: "Unknown",
            1: "Accident",
            2: "Fog",
            3: "Dangerous Conditions",
            4: "Rain",
            5: "Ice",
            6: "Jam",
            7: "Lane Closed",
            8: "Road Closed",
            9: "Road Works",
            10: "Wind",
            11: "Flooding",
            14: "Broken Down Vehicle",
        }

        for inc in incidents_raw:
            props = inc.get("properties", {})
            events = props.get("events", [])
            description = events[0].get("description", "") if events else ""

            category_id = props.get("iconCategory", 0)

            # Extract coordinates for location context
            geometry = inc.get("geometry", {})
            coords = geometry.get("coordinates", [])
            start_coord = None
            if coords and isinstance(coords, list):
                # Could be LineString (list of points) or Point
                if isinstance(coords[0], list):
                    start_coord = coords[0]  # First point of line
                else:
                    start_coord = coords

            incidents.append({
                "id": props.get("id", ""),
                "category": category_map.get(category_id, "Other"),
                "category_id": category_id,
                "description": description,
                "from_road": props.get("from", ""),
                "to_road": props.get("to", ""),
                "road_numbers": props.get("roadNumbers", []),
                "delay_seconds": props.get("delay", 0),
                "magnitude": props.get("magnitudeOfDelay", 0),
                "length_meters": props.get("length", 0),
                "start_time": props.get("startTime"),
                "end_time": props.get("endTime"),
                "coordinates": start_coord,
            })

        return incidents

    def _count_by_category(self, incidents: list[dict]) -> dict[str, int]:
        """Count incidents by category."""
        counts: dict[str, int] = {}
        for inc in incidents:
            cat = inc["category"]
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _build_incidents_summary(self, incidents: list[dict]) -> str:
        """Build human-readable incidents summary."""
        total = len(incidents)
        if total == 0:
            return "No traffic incidents currently reported in the area."

        by_cat = self._count_by_category(incidents)
        cat_str = ", ".join(f"{count} {cat}" for cat, count in by_cat.items())

        parts = [f"{total} traffic incidents in the area ({cat_str})."]

        # Show top 5 most significant (by delay)
        sorted_inc = sorted(incidents, key=lambda x: x.get("delay_seconds", 0), reverse=True)
        for inc in sorted_inc[:5]:
            desc = inc["description"][:80] if inc["description"] else inc["category"]
            from_road = inc["from_road"][:40] if inc["from_road"] else "Unknown location"
            delay = inc["delay_seconds"]
            delay_str = f" (delay: {delay}s)" if delay > 0 else ""
            parts.append(f"  - [{inc['category']}] {from_road}: {desc}{delay_str}")

        if total > 5:
            parts.append(f"  ... and {total - 5} more.")

        return "\n".join(parts)

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _get_location_label(latitude: float, longitude: float) -> str:
        """Try to match lat/lon to a known London location."""
        for key, info in LONDON_POINTS.items():
            if (
                abs(latitude - info["lat"]) < 0.005
                and abs(longitude - info["lon"]) < 0.005
            ):
                return info["label"]
        return f"({latitude:.4f}, {longitude:.4f})"

    @staticmethod
    def get_available_points() -> dict:
        """Return all predefined London points for the agent to use."""
        return {
            key: {"lat": info["lat"], "lon": info["lon"], "label": info["label"]}
            for key, info in LONDON_POINTS.items()
        }

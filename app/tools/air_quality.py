"""
Smart City AI Agent - OpenAQ Air Quality Tool
Fetches air quality measurements from monitoring stations.

OpenAQ API v3: https://api.openaq.org/v3 (free, no key required)
Provides: PM2.5, PM10, NO2, O3, SO2, CO readings from nearby stations.
"""

import logging
from datetime import datetime, timezone

from app.config import get_settings
from app.tools.base import BaseTool
from app.models.schemas import ToolResponse

logger = logging.getLogger(__name__)

# ── London Default Coordinates ────────────────────────────────────
LONDON_LAT = 51.5074
LONDON_LON = -0.1278

# ── AQI Breakpoints for PM2.5 (µg/m³) ───────────────────────────
# Simplified US EPA AQI categories for PM2.5
PM25_AQI_LEVELS = [
    (0, 12.0, "Good", "Air quality is satisfactory."),
    (12.1, 35.4, "Moderate", "Acceptable. Sensitive groups may experience minor effects."),
    (35.5, 55.4, "Unhealthy for Sensitive Groups", "Sensitive groups should reduce outdoor activity."),
    (55.5, 150.4, "Unhealthy", "Everyone may begin to experience health effects."),
    (150.5, 250.4, "Very Unhealthy", "Health alert: significant risk for everyone."),
    (250.5, 999, "Hazardous", "Emergency conditions. Avoid all outdoor activity."),
]

# ── Key Pollutant Parameters ──────────────────────────────────────
POLLUTANT_NAMES = {
    "pm25": "PM2.5 (Fine Particles)",
    "pm10": "PM10 (Coarse Particles)",
    "no2": "NO₂ (Nitrogen Dioxide)",
    "o3": "O₃ (Ozone)",
    "so2": "SO₂ (Sulfur Dioxide)",
    "co": "CO (Carbon Monoxide)",
}


class AirQualityTool(BaseTool):
    """
    OpenAQ air quality data tool.
    Fetches latest pollution readings from monitoring stations near a location.
    """

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self._base_url = settings.OPENAQ_BASE_URL
        self._api_key = settings.OPENAQ_API_KEY
        self._headers = {"X-API-Key": self._api_key} if self._api_key else {}

    @property
    def name(self) -> str:
        return "air_quality"

    @property
    def description(self) -> str:
        return (
            "Air quality data tool using OpenAQ. "
            "Fetches latest pollution readings (PM2.5, PM10, NO2, O3, SO2, CO) "
            "from monitoring stations near a given location. "
            "Includes AQI category interpretation. "
            "Defaults to Central London. Use this when the user asks about "
            "air quality, pollution, smog, or when correlating air quality "
            "with traffic or weather conditions."
        )

    def get_capabilities(self) -> list[str]:
        return ["latest_readings", "nearby_stations"]

    # ── Nearby Stations ───────────────────────────────────────────

    def get_nearby_stations(
        self,
        latitude: float = LONDON_LAT,
        longitude: float = LONDON_LON,
        radius_meters: int = 10000,
        limit: int = 5,
    ) -> ToolResponse:
        """
        Find air quality monitoring stations near a location.
        Args:
            latitude: Location latitude (default: Central London)
            longitude: Location longitude (default: Central London)
            radius_meters: Search radius in meters (default: 10km)
            limit: Max stations to return (default: 5)
        """
        url = f"{self._base_url}/locations"
        params = {
            "coordinates": f"{latitude},{longitude}",
            "radius": radius_meters,
            "limit": limit,
        }
        query_type = "nearby_stations"

        try:
            raw_data, elapsed_ms = self._timed_request(url, params=params, headers=self._headers)
            stations = self._parse_stations(raw_data)
            summary = self._build_stations_summary(
                stations, latitude, longitude, radius_meters
            )

            return ToolResponse(
                tool_name=self.name,
                query_type=query_type,
                success=True,
                data={
                    "stations": stations,
                    "total_found": len(stations),
                    "search_radius_m": radius_meters,
                },
                summary=summary,
                timestamp=datetime.now(tz=timezone.utc),
                source_url=url,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Failed to fetch nearby stations: {e}")
            return self._build_error_response(query_type, e, url)

    def _parse_stations(self, raw_data: dict) -> list[dict]:
        """Parse OpenAQ locations response."""
        results = raw_data.get("results", [])
        stations = []
        for loc in results:
            coordinates = loc.get("coordinates", {})
            stations.append({
                "id": loc.get("id"),
                "name": loc.get("name", "Unknown"),
                "locality": loc.get("locality", ""),
                "latitude": coordinates.get("latitude"),
                "longitude": coordinates.get("longitude"),
                "is_monitor": loc.get("isMonitor", False),
                "parameters": [
                    p.get("parameter", "") for p in loc.get("parameters", [])
                ],
                "last_updated": loc.get("datetimeLast", {}).get("utc"),
            })
        return stations

    def _build_stations_summary(
        self,
        stations: list[dict],
        latitude: float,
        longitude: float,
        radius: int,
    ) -> str:
        """Build summary of nearby monitoring stations."""
        location = self._get_location_label(latitude, longitude)
        if not stations:
            return (
                f"No air quality monitoring stations found within "
                f"{radius / 1000:.0f}km of {location}."
            )

        parts = [
            f"Found {len(stations)} monitoring station(s) near {location}:"
        ]
        for s in stations:
            name = s["name"]
            params = ", ".join(s["parameters"][:4]) if s["parameters"] else "N/A"
            parts.append(f"  - {name}: measures [{params}]")

        return "\n".join(parts)

    # ── Latest Readings ───────────────────────────────────────────

    def get_latest_readings(
        self,
        latitude: float = LONDON_LAT,
        longitude: float = LONDON_LON,
        radius_meters: int = 10000,
        limit: int = 5,
    ) -> ToolResponse:
        """
        Fetch latest air quality readings from nearby stations.
        Finds stations, then gets their most recent measurements.
        Args:
            latitude: Location latitude (default: Central London)
            longitude: Location longitude (default: Central London)
            radius_meters: Search radius in meters (default: 10km)
            limit: Max stations to query (default: 5)
        """
        query_type = "latest_readings"

        try:
            # Step 1: Find nearby stations
            stations_response = self.get_nearby_stations(
                latitude, longitude, radius_meters, limit
            )

            if not stations_response.success:
                return self._build_error_response(
                    query_type,
                    Exception(stations_response.error or "Failed to find stations"),
                )

            stations = stations_response.data.get("stations", [])
            if not stations:
                return ToolResponse(
                    tool_name=self.name,
                    query_type=query_type,
                    success=True,
                    data={"readings": [], "stations_checked": 0},
                    summary=(
                        f"No monitoring stations found within "
                        f"{radius_meters / 1000:.0f}km. No air quality data available."
                    ),
                    timestamp=datetime.now(tz=timezone.utc),
                )

            # Step 2: Get latest readings from each station
            all_readings = []
            for station in stations:
                station_id = station.get("id")
                if not station_id:
                    continue

                readings = self._fetch_station_readings(station_id, station["name"])
                all_readings.extend(readings)

            # Step 3: Aggregate by pollutant (latest reading per pollutant)
            aggregated = self._aggregate_readings(all_readings)
            pm25_value = aggregated.get("pm25", {}).get("value")
            aqi_info = self._get_aqi_category(pm25_value) if pm25_value else None

            summary = self._build_readings_summary(
                aggregated, aqi_info, latitude, longitude
            )

            return ToolResponse(
                tool_name=self.name,
                query_type=query_type,
                success=True,
                data={
                    "readings": aggregated,
                    "aqi_category": aqi_info,
                    "stations_checked": len(stations),
                    "raw_reading_count": len(all_readings),
                },
                summary=summary,
                timestamp=datetime.now(tz=timezone.utc),
                source_url=f"{self._base_url}/locations",
                response_time_ms=stations_response.response_time_ms,
            )

        except Exception as e:
            logger.error(f"Failed to fetch latest readings: {e}")
            return self._build_error_response(query_type, e)

    def _fetch_station_readings(
        self, station_id: int, station_name: str
    ) -> list[dict]:
        """Fetch latest sensor readings for a specific station."""
        url = f"{self._base_url}/locations/{station_id}/latest"
        try:
            raw_data, _ = self._timed_request(url, headers=self._headers)
            results = raw_data.get("results", [])

            readings = []
            for r in results:
                param = r.get("parameter", {})
                period = r.get("period", {})
                readings.append({
                    "parameter": param.get("name", "unknown"),
                    "display_name": param.get("displayName", param.get("name", "")),
                    "value": r.get("value"),
                    "unit": param.get("units", ""),
                    "station_name": station_name,
                    "station_id": station_id,
                    "last_updated": period.get("datetimeTo", {}).get("utc") if period else None,
                })
            return readings

        except Exception as e:
            logger.warning(f"Failed to fetch readings for station {station_id}: {e}")
            return []

    def _aggregate_readings(self, readings: list[dict]) -> dict:
        """
        Aggregate readings: keep the most recent reading per pollutant.
        Returns dict keyed by parameter name.
        """
        latest: dict[str, dict] = {}
        for r in readings:
            param = r["parameter"]
            # Keep the reading with a value (prefer non-None)
            if param not in latest or (
                r["value"] is not None and latest[param]["value"] is None
            ):
                latest[param] = r
        return latest

    def _get_aqi_category(self, pm25_value: float) -> dict | None:
        """Determine AQI category based on PM2.5 value."""
        for low, high, category, description in PM25_AQI_LEVELS:
            if low <= pm25_value <= high:
                return {
                    "category": category,
                    "description": description,
                    "pm25_value": pm25_value,
                }
        return None

    def _build_readings_summary(
        self,
        aggregated: dict,
        aqi_info: dict | None,
        latitude: float,
        longitude: float,
    ) -> str:
        """Build human-readable air quality summary."""
        location = self._get_location_label(latitude, longitude)

        if not aggregated:
            return f"No recent air quality readings available near {location}."

        parts = [f"Air quality near {location}:"]

        # AQI headline if available
        if aqi_info:
            parts.append(
                f"Overall: {aqi_info['category']} — {aqi_info['description']}"
            )

        # Individual pollutant readings
        for param_key, reading in aggregated.items():
            display = POLLUTANT_NAMES.get(param_key, reading.get("display_name", param_key))
            value = reading.get("value")
            unit = reading.get("unit", "")
            if value is not None:
                parts.append(f"  - {display}: {value} {unit}")

        return "\n".join(parts)

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _get_location_label(latitude: float, longitude: float) -> str:
        """Return 'Central London' for default coords, else lat/lon string."""
        if (
            abs(latitude - LONDON_LAT) < 0.01
            and abs(longitude - LONDON_LON) < 0.01
        ):
            return "Central London"
        return f"({latitude:.4f}, {longitude:.4f})"

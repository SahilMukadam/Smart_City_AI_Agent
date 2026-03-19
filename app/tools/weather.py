"""
Smart City AI Agent - Open-Meteo Weather Tool
Fetches current weather conditions and short-term forecasts.

Open-Meteo API: https://open-meteo.com/en/docs (free, no key required)
Provides: temperature, humidity, wind, precipitation, weather codes.
"""

import logging
from datetime import datetime, timezone

from app.config import get_settings
from app.tools.base import BaseTool
from app.models.schemas import ToolResponse

logger = logging.getLogger(__name__)

# ── Weather Code Descriptions (WMO standard) ─────────────────────
# Open-Meteo returns WMO weather interpretation codes.
# Full list: https://open-meteo.com/en/docs#weathervariables
WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# ── London Default Coordinates ────────────────────────────────────
LONDON_LAT = 51.5074
LONDON_LON = -0.1278


class WeatherTool(BaseTool):
    """
    Open-Meteo weather data tool.
    Provides current conditions and short-term forecast for any location.
    Defaults to Central London if no coordinates specified.
    """

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self._base_url = settings.OPEN_METEO_BASE_URL

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return (
            "Weather data tool using Open-Meteo. "
            "Fetches current weather conditions (temperature, humidity, wind, "
            "precipitation, weather description) and short-term hourly forecast. "
            "Defaults to Central London. Use this when the user asks about weather, "
            "rain, temperature, or when correlating weather with traffic/air quality."
        )

    def get_capabilities(self) -> list[str]:
        return ["current_weather", "forecast"]

    # ── Current Weather ───────────────────────────────────────────

    def get_current_weather(
        self,
        latitude: float = LONDON_LAT,
        longitude: float = LONDON_LON,
    ) -> ToolResponse:
        """
        Fetch current weather conditions for a location.
        Args:
            latitude: Location latitude (default: Central London)
            longitude: Location longitude (default: Central London)
        """
        url = f"{self._base_url}/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "cloud_cover",
                "pressure_msl",
                "visibility",
            ]),
            "timezone": "Europe/London",
        }
        query_type = "current_weather"

        try:
            raw_data, elapsed_ms = self._timed_request(url, params=params)
            current = raw_data.get("current", {})
            parsed = self._parse_current(current)
            summary = self._build_current_summary(parsed, latitude, longitude)

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
            logger.error(f"Failed to fetch current weather: {e}")
            return self._build_error_response(query_type, e, url)

    def _parse_current(self, current: dict) -> dict:
        """Parse current weather data into a clean dictionary."""
        weather_code = current.get("weather_code", -1)
        return {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation", 0),
            "weather_code": weather_code,
            "weather_description": WEATHER_CODES.get(weather_code, "Unknown"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_deg": current.get("wind_direction_10m"),
            "wind_gusts_kmh": current.get("wind_gusts_10m"),
            "cloud_cover_percent": current.get("cloud_cover"),
            "pressure_hpa": current.get("pressure_msl"),
            "visibility_m": current.get("visibility"),
            "time": current.get("time"),
        }

    def _build_current_summary(
        self,
        parsed: dict,
        latitude: float,
        longitude: float,
    ) -> str:
        """Build human-readable current weather summary."""
        location = self._get_location_label(latitude, longitude)
        description = parsed.get("weather_description", "Unknown")
        temp = parsed.get("temperature_c", "N/A")
        feels_like = parsed.get("feels_like_c", "N/A")
        humidity = parsed.get("humidity_percent", "N/A")
        wind = parsed.get("wind_speed_kmh", "N/A")
        precip = parsed.get("precipitation_mm", 0)

        parts = [
            f"Current weather in {location}: {description}.",
            f"Temperature: {temp}°C (feels like {feels_like}°C).",
            f"Humidity: {humidity}%. Wind: {wind} km/h.",
        ]
        if precip and precip > 0:
            parts.append(f"Precipitation: {precip} mm.")

        return " ".join(parts)

    # ── Forecast ──────────────────────────────────────────────────

    def get_forecast(
        self,
        latitude: float = LONDON_LAT,
        longitude: float = LONDON_LON,
        hours: int = 12,
    ) -> ToolResponse:
        """
        Fetch hourly weather forecast.
        Args:
            latitude: Location latitude (default: Central London)
            longitude: Location longitude (default: Central London)
            hours: Number of forecast hours (default: 12, max: 48)
        """
        hours = min(hours, 48)  # Cap at 48h
        url = f"{self._base_url}/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join([
                "temperature_2m",
                "precipitation_probability",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "visibility",
            ]),
            "timezone": "Europe/London",
            "forecast_hours": hours,
        }
        query_type = "forecast"

        try:
            raw_data, elapsed_ms = self._timed_request(url, params=params)
            hourly = raw_data.get("hourly", {})
            parsed = self._parse_forecast(hourly, hours)
            summary = self._build_forecast_summary(parsed, latitude, longitude, hours)

            return ToolResponse(
                tool_name=self.name,
                query_type=query_type,
                success=True,
                data={
                    "hours_requested": hours,
                    "hourly": parsed,
                },
                summary=summary,
                timestamp=datetime.now(tz=timezone.utc),
                source_url=url,
                response_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Failed to fetch forecast: {e}")
            return self._build_error_response(query_type, e, url)

    def _parse_forecast(self, hourly: dict, hours: int) -> list[dict]:
        """Parse hourly forecast into a list of hour-by-hour data."""
        times = hourly.get("time", [])[:hours]
        temps = hourly.get("temperature_2m", [])[:hours]
        precip_probs = hourly.get("precipitation_probability", [])[:hours]
        precips = hourly.get("precipitation", [])[:hours]
        codes = hourly.get("weather_code", [])[:hours]
        winds = hourly.get("wind_speed_10m", [])[:hours]
        visibilities = hourly.get("visibility", [])[:hours]

        result = []
        for i in range(len(times)):
            code = codes[i] if i < len(codes) else -1
            result.append({
                "time": times[i],
                "temperature_c": temps[i] if i < len(temps) else None,
                "precipitation_probability": precip_probs[i] if i < len(precip_probs) else None,
                "precipitation_mm": precips[i] if i < len(precips) else None,
                "weather_code": code,
                "weather_description": WEATHER_CODES.get(code, "Unknown"),
                "wind_speed_kmh": winds[i] if i < len(winds) else None,
                "visibility_m": visibilities[i] if i < len(visibilities) else None,
            })
        return result

    def _build_forecast_summary(
        self,
        hourly: list[dict],
        latitude: float,
        longitude: float,
        hours: int,
    ) -> str:
        """Build human-readable forecast summary with key highlights."""
        if not hourly:
            return f"No forecast data available for the next {hours} hours."

        location = self._get_location_label(latitude, longitude)

        # Find key stats
        temps = [h["temperature_c"] for h in hourly if h["temperature_c"] is not None]
        rain_probs = [
            h["precipitation_probability"]
            for h in hourly
            if h["precipitation_probability"] is not None
        ]
        rain_hours = [
            h for h in hourly
            if h.get("precipitation_mm") and h["precipitation_mm"] > 0
        ]

        parts = [f"{hours}-hour forecast for {location}:"]

        if temps:
            parts.append(
                f"Temperature range: {min(temps):.1f}°C to {max(temps):.1f}°C."
            )

        if rain_probs:
            max_rain_prob = max(rain_probs)
            if max_rain_prob > 50:
                parts.append(f"Rain likely (up to {max_rain_prob}% chance).")
            elif max_rain_prob > 20:
                parts.append(f"Some chance of rain (up to {max_rain_prob}%).")
            else:
                parts.append("Rain unlikely.")

        if rain_hours:
            total_precip = sum(h["precipitation_mm"] for h in rain_hours)
            parts.append(
                f"{len(rain_hours)} hours with precipitation (total: {total_precip:.1f} mm)."
            )

        return " ".join(parts)

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

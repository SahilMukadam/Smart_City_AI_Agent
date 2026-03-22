"""
Smart City AI Agent - LangChain Tool Definitions
Wraps each data source tool into a LangChain-compatible tool
that the LangGraph agent can invoke.

Each tool has a clear description so the LLM knows WHEN to use it.
"""

import json
import logging

from langchain_core.tools import tool

from app.tools.tfl import TfLTool
from app.tools.weather import WeatherTool
from app.tools.air_quality import AirQualityTool
from app.tools.tomtom import TomTomTool

logger = logging.getLogger(__name__)

# ── Singleton tool instances ──────────────────────────────────────
_tfl = TfLTool()
_weather = WeatherTool()
_air_quality = AirQualityTool()
_tomtom = TomTomTool()


# ══════════════════════════════════════════════════════════════════
# TfL Tools
# ══════════════════════════════════════════════════════════════════

@tool
def get_tube_status() -> str:
    """
    Get the current status of all London Underground tube lines.
    Use this when the user asks about tube delays, underground service,
    or general London transport disruptions.
    Returns status of all lines with disruption reasons if any.
    """
    result = _tfl.get_tube_status()
    return result.to_agent_string()


@tool
def get_road_disruptions() -> str:
    """
    Get all current road disruptions across London including roadworks,
    incidents, and closures from Transport for London (TfL).
    Use this when the user asks about road closures, roadworks,
    or TfL-reported disruptions.
    Returns list of disruptions with severity and location.
    """
    result = _tfl.get_road_disruptions()
    return result.to_agent_string()


@tool
def get_road_corridor_status(road_ids: str = "") -> str:
    """
    Get the status of specific major road corridors in London.
    Args:
        road_ids: Comma-separated road IDs like "A1,A2,A40".
                  Leave empty for all major roads.
    Use this when the user asks about specific road conditions
    or wants an overview of major road status.
    """
    result = _tfl.get_road_status(road_ids=road_ids if road_ids else None)
    return result.to_agent_string()


# ══════════════════════════════════════════════════════════════════
# Weather Tools
# ══════════════════════════════════════════════════════════════════

@tool
def get_current_weather(latitude: float = 51.5074, longitude: float = -0.1278) -> str:
    """
    Get current weather conditions for a location.
    Args:
        latitude: Location latitude (default: Central London 51.5074)
        longitude: Location longitude (default: Central London -0.1278)
    Returns temperature, humidity, wind, precipitation, and weather description.
    Use this when the user asks about current weather, rain, temperature,
    or when you need weather context to explain traffic or air quality patterns.
    """
    result = _weather.get_current_weather(latitude=latitude, longitude=longitude)
    return result.to_agent_string()


@tool
def get_weather_forecast(
    latitude: float = 51.5074,
    longitude: float = -0.1278,
    hours: int = 12,
) -> str:
    """
    Get hourly weather forecast for a location.
    Args:
        latitude: Location latitude (default: Central London 51.5074)
        longitude: Location longitude (default: Central London -0.1278)
        hours: Number of forecast hours, 1-48 (default: 12)
    Returns temperature range, rain probability, and precipitation forecast.
    Use this when the user asks about upcoming weather, whether it will rain,
    or needs a forecast for planning.
    """
    result = _weather.get_forecast(
        latitude=latitude, longitude=longitude, hours=min(hours, 48)
    )
    return result.to_agent_string()


# ══════════════════════════════════════════════════════════════════
# Air Quality Tools
# ══════════════════════════════════════════════════════════════════

@tool
def get_air_quality(
    latitude: float = 51.5074,
    longitude: float = -0.1278,
) -> str:
    """
    Get latest air quality readings near a location.
    Args:
        latitude: Location latitude (default: Central London 51.5074)
        longitude: Location longitude (default: Central London -0.1278)
    Returns PM2.5, PM10, NO2, O3 readings with AQI category
    (Good/Moderate/Unhealthy/etc).
    Use this when the user asks about air quality, pollution, smog,
    or when correlating pollution with traffic or weather.
    """
    result = _air_quality.get_latest_readings(
        latitude=latitude, longitude=longitude
    )
    return result.to_agent_string()


# ══════════════════════════════════════════════════════════════════
# TomTom Traffic Tools
# ══════════════════════════════════════════════════════════════════

@tool
def get_traffic_flow(
    latitude: float = 51.5074,
    longitude: float = -0.1278,
    location_name: str = "",
) -> str:
    """
    Get real-time traffic flow data (speed, congestion level) for a road
    segment near a point.
    Args:
        latitude: Location latitude (default: Central London 51.5074)
        longitude: Location longitude (default: Central London -0.1278)
        location_name: Optional human-readable name (e.g., "Oxford Street")
    Returns current speed, free-flow speed, congestion ratio and level.
    Use this when the user asks about traffic speed, congestion at a specific
    location, or real-time road conditions.
    """
    result = _tomtom.get_traffic_flow(
        latitude=latitude,
        longitude=longitude,
        location_name=location_name if location_name else None,
    )
    return result.to_agent_string()


@tool
def get_london_traffic_overview() -> str:
    """
    Get traffic flow at multiple key London locations at once.
    Checks 10 predefined points: Central London, City of London,
    Westminster, Camden, Tower Bridge, King's Cross, Canary Wharf,
    Shoreditch, Brixton, and Hammersmith.
    Results are sorted worst-congestion-first.
    Use this when the user asks for a general London traffic overview
    or wants to know which areas have the worst congestion.
    """
    result = _tomtom.get_multi_point_flow()
    return result.to_agent_string()


@tool
def get_traffic_incidents() -> str:
    """
    Get current traffic incidents (accidents, roadworks, closures, jams)
    across Greater London from TomTom.
    Returns incidents sorted by severity/delay with location and description.
    Use this when the user asks about accidents, incidents, or wants to
    know WHY traffic is bad in a specific area.
    """
    result = _tomtom.get_traffic_incidents()
    return result.to_agent_string()


# ══════════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    get_tube_status,
    get_road_disruptions,
    get_road_corridor_status,
    get_current_weather,
    get_weather_forecast,
    get_air_quality,
    get_traffic_flow,
    get_london_traffic_overview,
    get_traffic_incidents,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def get_tool_descriptions() -> str:
    """
    Return a formatted string of all available tools and their descriptions.
    Useful for debugging and for the system prompt.
    """
    lines = []
    for t in ALL_TOOLS:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)

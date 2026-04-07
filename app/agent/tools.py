"""
Smart City AI Agent - LangChain Tool Definitions (Day 13)
All tools use centralized config for default coordinates.
Adds geocode_location tool for resolving any London address.
"""

import logging
from langchain_core.tools import tool
from app.config import get_settings
from app.tools.tfl import TfLTool
from app.tools.weather import WeatherTool
from app.tools.air_quality import AirQualityTool
from app.tools.tomtom import TomTomTool
from app.tools.geocoder import geocode

logger = logging.getLogger(__name__)

_tfl = TfLTool()
_weather = WeatherTool()
_air_quality = AirQualityTool()
_tomtom = TomTomTool()


# ══════════════════════════════════════════════════════════════════
# Geocoder Tool
# ══════════════════════════════════════════════════════════════════

@tool
def geocode_location(query: str) -> str:
    """
    Convert a place name, street, or address in London to coordinates.
    Args:
        query: Any London location (e.g., "Baker Street", "Brick Lane",
               "London Bridge", "Shoreditch High Street", "Greenwich Park")
    Returns coordinates that can be used with other tools.
    Use this when the user mentions a specific street, area, or landmark
    that isn't in the predefined list of monitoring points.
    """
    result = geocode(f"{query}, London")
    if result:
        return (
            f"Geocoded '{query}': latitude={result['latitude']:.6f}, "
            f"longitude={result['longitude']:.6f} ({result['display_name']})"
        )
    return f"Could not find coordinates for '{query}' in London."


# ══════════════════════════════════════════════════════════════════
# TfL Tools
# ══════════════════════════════════════════════════════════════════

@tool
def get_tube_status() -> str:
    """
    Get the current status of all London Underground tube lines.
    Use when the user asks about tube delays, underground service,
    or general London transport disruptions.
    """
    return _tfl.get_tube_status().to_agent_string()


@tool
def get_road_disruptions() -> str:
    """
    Get all current road disruptions across London including roadworks,
    incidents, and closures from Transport for London (TfL).
    """
    return _tfl.get_road_disruptions().to_agent_string()


@tool
def get_road_corridor_status(road_ids: str = "") -> str:
    """
    Get the status of specific major road corridors in London.
    Args:
        road_ids: Comma-separated road IDs like "A1,A2,A40". Leave empty for all.
    """
    return _tfl.get_road_status(road_ids=road_ids if road_ids else None).to_agent_string()


# ══════════════════════════════════════════════════════════════════
# Weather Tools
# ══════════════════════════════════════════════════════════════════

@tool
def get_current_weather(latitude: float = 0.0, longitude: float = 0.0) -> str:
    """
    Get current weather conditions for a location.
    Args:
        latitude: Location latitude (0 = use default Central London)
        longitude: Location longitude (0 = use default Central London)
    Returns temperature, humidity, wind, precipitation, and weather description.
    """
    settings = get_settings()
    lat = latitude if latitude != 0.0 else settings.DEFAULT_LATITUDE
    lon = longitude if longitude != 0.0 else settings.DEFAULT_LONGITUDE
    return _weather.get_current_weather(latitude=lat, longitude=lon).to_agent_string()


@tool
def get_weather_forecast(latitude: float = 0.0, longitude: float = 0.0, hours: int = 12) -> str:
    """
    Get hourly weather forecast for a location.
    Args:
        latitude: Location latitude (0 = use default Central London)
        longitude: Location longitude (0 = use default Central London)
        hours: Number of forecast hours, 1-48 (default: 12)
    """
    settings = get_settings()
    lat = latitude if latitude != 0.0 else settings.DEFAULT_LATITUDE
    lon = longitude if longitude != 0.0 else settings.DEFAULT_LONGITUDE
    return _weather.get_forecast(latitude=lat, longitude=lon, hours=min(hours, 48)).to_agent_string()


# ══════════════════════════════════════════════════════════════════
# Air Quality Tools
# ══════════════════════════════════════════════════════════════════

@tool
def get_air_quality(latitude: float = 0.0, longitude: float = 0.0) -> str:
    """
    Get latest air quality readings near a location.
    Args:
        latitude: Location latitude (0 = use default Central London)
        longitude: Location longitude (0 = use default Central London)
    Returns PM2.5, PM10, NO2, O3 readings with AQI category.
    """
    settings = get_settings()
    lat = latitude if latitude != 0.0 else settings.DEFAULT_LATITUDE
    lon = longitude if longitude != 0.0 else settings.DEFAULT_LONGITUDE
    return _air_quality.get_latest_readings(latitude=lat, longitude=lon).to_agent_string()


# ══════════════════════════════════════════════════════════════════
# TomTom Traffic Tools
# ══════════════════════════════════════════════════════════════════

@tool
def get_traffic_flow(latitude: float = 0.0, longitude: float = 0.0, location_name: str = "") -> str:
    """
    Get real-time traffic flow data (speed, congestion level) near a point.
    Args:
        latitude: Location latitude (0 = use default Central London)
        longitude: Location longitude (0 = use default Central London)
        location_name: Optional human-readable name (e.g., "Baker Street")
    """
    settings = get_settings()
    lat = latitude if latitude != 0.0 else settings.DEFAULT_LATITUDE
    lon = longitude if longitude != 0.0 else settings.DEFAULT_LONGITUDE
    return _tomtom.get_traffic_flow(
        latitude=lat, longitude=lon,
        location_name=location_name if location_name else None,
    ).to_agent_string()


@tool
def get_london_traffic_overview() -> str:
    """
    Get traffic flow at multiple key London locations at once.
    Results sorted worst-congestion-first. Use for a general London traffic overview.
    """
    return _tomtom.get_multi_point_flow().to_agent_string()


@tool
def get_traffic_incidents() -> str:
    """
    Get current traffic incidents (accidents, roadworks, closures, jams)
    across Greater London from TomTom.
    """
    return _tomtom.get_traffic_incidents().to_agent_string()


# ══════════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    geocode_location,
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
    return "\n".join(f"- {t.name}: {t.description}" for t in ALL_TOOLS)

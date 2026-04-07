"""
Smart City AI Agent - Nominatim Geocoder
Resolves place names and addresses to coordinates using OpenStreetMap.
Free, no API key required. Constrained to London bounding box.

Usage:
    from app.tools.geocoder import geocode
    result = geocode("Baker Street, London")
    # {"latitude": 51.5208, "longitude": -0.1570, "display_name": "Baker Street, Marylebone, ..."}
"""

import logging
import time
from threading import Lock

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── In-memory geocode cache ───────────────────────────────────────
_cache: dict[str, dict | None] = {}
_cache_lock = Lock()
_last_request_time: float = 0
_rate_lock = Lock()

# Nominatim requires a descriptive User-Agent
USER_AGENT = "SmartCityAIAgent/1.0 (portfolio project; contact: s.mukadamwrk@gmail.com)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def geocode(
    query: str,
    constrain_to_bbox: bool = True,
) -> dict | None:
    """
    Geocode a place name or address to coordinates.

    Args:
        query: Place name, street, or address (e.g., "Baker Street, London")
        constrain_to_bbox: If True, limits results to the configured bounding box

    Returns:
        Dict with latitude, longitude, display_name, or None if not found.
    """
    settings = get_settings()

    # Normalize query for cache key
    cache_key = query.strip().lower()

    # Check cache
    with _cache_lock:
        if cache_key in _cache:
            logger.info(f"📍 Geocode cache hit: {query}")
            return _cache[cache_key]

    # Rate limit: max 1 request per second (Nominatim policy)
    _enforce_rate_limit()

    # Build request
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }

    # Constrain to London bounding box
    if constrain_to_bbox:
        params["viewbox"] = (
            f"{settings.BBOX_MIN_LON},{settings.BBOX_MAX_LAT},"
            f"{settings.BBOX_MAX_LON},{settings.BBOX_MIN_LAT}"
        )
        params["bounded"] = 1

    headers = {"User-Agent": USER_AGENT}

    try:
        response = httpx.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()

        if not results:
            logger.info(f"📍 Geocode no results: {query}")
            with _cache_lock:
                _cache[cache_key] = None
            return None

        first = results[0]
        result = {
            "latitude": float(first["lat"]),
            "longitude": float(first["lon"]),
            "display_name": first.get("display_name", query),
        }

        # Cache the result
        with _cache_lock:
            _cache[cache_key] = result

        logger.info(
            f"📍 Geocoded: {query} → ({result['latitude']:.4f}, {result['longitude']:.4f})"
        )
        return result

    except Exception as e:
        logger.error(f"📍 Geocode error for '{query}': {e}")
        return None


def geocode_if_needed(tool_args: dict) -> dict:
    """
    Post-process tool arguments: if a tool has location_name but
    default/missing coordinates, geocode the location name.

    Called by the argument extractor after LLM produces the JSON.
    """
    settings = get_settings()
    default_lat = settings.DEFAULT_LATITUDE
    default_lon = settings.DEFAULT_LONGITUDE

    updated = {}
    for tool_key, args in tool_args.items():
        if not isinstance(args, dict):
            updated[tool_key] = args
            continue

        args = dict(args)  # Don't mutate original
        location_name = args.get("location_name", "")

        # Check if coordinates are default/missing and we have a location name
        has_custom_coords = (
            "latitude" in args
            and "longitude" in args
            and not (
                abs(args["latitude"] - default_lat) < 0.001
                and abs(args["longitude"] - default_lon) < 0.001
            )
        )

        if location_name and not has_custom_coords:
            # Try to geocode
            geo = geocode(f"{location_name}, London")
            if geo:
                args["latitude"] = geo["latitude"]
                args["longitude"] = geo["longitude"]
                logger.info(f"🎯 Geocoded {location_name} for {tool_key}")

        updated[tool_key] = args

    return updated


def clear_geocode_cache():
    """Clear the geocode cache."""
    with _cache_lock:
        _cache.clear()


def _enforce_rate_limit():
    """Ensure at least 1 second between Nominatim requests."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _last_request_time = time.monotonic()

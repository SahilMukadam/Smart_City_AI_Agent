"""
Smart City AI Agent - Configuration Management
Single source of truth for all settings including location defaults.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # ── App Settings ──────────────────────────────────────────────
    APP_NAME: str = "Smart City AI Agent"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── Default Location (Central London) ─────────────────────────
    # SINGLE SOURCE OF TRUTH for all default coordinates.
    # Change these to adapt the agent for any city.
    DEFAULT_LATITUDE: float = 51.5074
    DEFAULT_LONGITUDE: float = -0.1278
    DEFAULT_LOCATION_NAME: str = "Central London"

    # ── Bounding Box (Greater London) ─────────────────────────────
    # Used for: TomTom incidents, geocoding constraints, map bounds
    BBOX_MIN_LAT: float = 51.28
    BBOX_MIN_LON: float = -0.51
    BBOX_MAX_LAT: float = 51.69
    BBOX_MAX_LON: float = 0.33

    # ── TfL API ───────────────────────────────────────────────────
    TFL_BASE_URL: str = "https://api.tfl.gov.uk"
    TFL_APP_KEY: str = ""

    # ── TomTom API ────────────────────────────────────────────────
    TOMTOM_API_KEY: str = ""
    TOMTOM_BASE_URL: str = "https://api.tomtom.com"

    # ── Open-Meteo ────────────────────────────────────────────────
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1"

    # ── OpenAQ ────────────────────────────────────────────────────
    OPENAQ_BASE_URL: str = "https://api.openaq.org/v3"
    OPENAQ_API_KEY: str = ""

    # ── Google Gemini ─────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"

    # ── Cache Settings ────────────────────────────────────────────
    CACHE_TTL_SECONDS: int = 300

    # ── HTTP Client ───────────────────────────────────────────────
    HTTP_TIMEOUT_SECONDS: int = 15
    HTTP_MAX_RETRIES: int = 2

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()

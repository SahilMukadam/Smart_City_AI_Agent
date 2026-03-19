"""
Smart City AI Agent - Configuration Management
Uses pydantic-settings for type-safe environment variable loading.
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

    # ── TfL API (free, no key required for basic access) ─────────
    TFL_BASE_URL: str = "https://api.tfl.gov.uk"
    TFL_APP_KEY: str = ""  # Optional: higher rate limits with key

    # ── TomTom API (Day 3) ───────────────────────────────────────
    TOMTOM_API_KEY: str = ""
    TOMTOM_BASE_URL: str = "https://api.tomtom.com"

    # ── Open-Meteo (Day 2, free, no key needed) ──────────────────
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1"

    # ── OpenAQ (Day 2, free, no key needed) ──────────────────────
    OPENAQ_BASE_URL: str = "https://api.openaq.org/v3"
    OPENAQ_API_KEY: str = ""

    # ── Google Gemini (Day 4) ────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # ── Cache Settings ───────────────────────────────────────────
    CACHE_TTL_SECONDS: int = 300  # 5 min default TTL for API responses

    # ── HTTP Client Settings ─────────────────────────────────────
    HTTP_TIMEOUT_SECONDS: int = 15
    HTTP_MAX_RETRIES: int = 2

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Call get_settings() anywhere to access config.
    The lru_cache ensures we only load .env once.
    """
    return Settings()

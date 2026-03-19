"""
Smart City AI Agent - FastAPI Application
Day 2: TfL + Weather + Air Quality endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.tools.tfl import TfLTool
from app.tools.weather import WeatherTool
from app.tools.air_quality import AirQualityTool

# ── Logging Setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan (startup / shutdown) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"TfL API key configured: {bool(settings.TFL_APP_KEY)}")
    yield
    logger.info("Shutting down.")


# ── App Instance ──────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="An autonomous AI agent for London city data analysis",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Tool Instances ────────────────────────────────────────────────
tfl_tool = TfLTool()
weather_tool = WeatherTool()
air_quality_tool = AirQualityTool()


# ── Health Check ──────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "0.2.0",
        "tools_available": ["tfl", "weather", "air_quality"],
    }


# ══════════════════════════════════════════════════════════════════
# TfL Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/tfl/tube-status")
def get_tube_status():
    """Fetch current status of all London Underground lines."""
    result = tfl_tool.get_tube_status()
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tfl/disruptions")
def get_disruptions():
    """Fetch all current road disruptions across London."""
    result = tfl_tool.get_road_disruptions()
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tfl/road-status")
def get_road_status(
    road_ids: str | None = Query(
        default=None,
        description="Comma-separated road IDs (e.g., 'A1,A2,A40'). Omit for all roads.",
    ),
):
    """Fetch status of major road corridors."""
    result = tfl_tool.get_road_status(road_ids=road_ids)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tfl/summary")
def get_tfl_summary():
    """Quick summary of all TfL data."""
    tube = tfl_tool.get_tube_status()
    disruptions = tfl_tool.get_road_disruptions()

    return {
        "tube_status": {
            "success": tube.success,
            "summary": tube.summary,
            "disrupted_count": tube.data.get("disrupted_count", 0) if tube.data else 0,
        },
        "road_disruptions": {
            "success": disruptions.success,
            "summary": disruptions.summary,
            "total_count": disruptions.data.get("total_count", 0) if disruptions.data else 0,
        },
    }


# ══════════════════════════════════════════════════════════════════
# Weather Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/weather/current")
def get_current_weather(
    lat: float = Query(default=51.5074, description="Latitude"),
    lon: float = Query(default=-0.1278, description="Longitude"),
):
    """Fetch current weather conditions."""
    result = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/weather/forecast")
def get_weather_forecast(
    lat: float = Query(default=51.5074, description="Latitude"),
    lon: float = Query(default=-0.1278, description="Longitude"),
    hours: int = Query(default=12, description="Forecast hours (max 48)", ge=1, le=48),
):
    """Fetch hourly weather forecast."""
    result = weather_tool.get_forecast(latitude=lat, longitude=lon, hours=hours)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/weather/summary")
def get_weather_summary(
    lat: float = Query(default=51.5074, description="Latitude"),
    lon: float = Query(default=-0.1278, description="Longitude"),
):
    """Quick summary: current weather + 6-hour forecast."""
    current = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    forecast = weather_tool.get_forecast(latitude=lat, longitude=lon, hours=6)

    return {
        "current": {
            "success": current.success,
            "summary": current.summary,
            "data": current.data if current.success else None,
        },
        "forecast_6h": {
            "success": forecast.success,
            "summary": forecast.summary,
        },
    }


# ══════════════════════════════════════════════════════════════════
# Air Quality Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/air-quality/stations")
def get_nearby_stations(
    lat: float = Query(default=51.5074, description="Latitude"),
    lon: float = Query(default=-0.1278, description="Longitude"),
    radius: int = Query(default=10000, description="Search radius in meters"),
):
    """Find air quality monitoring stations nearby."""
    result = air_quality_tool.get_nearby_stations(
        latitude=lat, longitude=lon, radius_meters=radius
    )
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/air-quality/latest")
def get_air_quality_latest(
    lat: float = Query(default=51.5074, description="Latitude"),
    lon: float = Query(default=-0.1278, description="Longitude"),
    radius: int = Query(default=10000, description="Search radius in meters"),
):
    """Fetch latest air quality readings from nearby stations."""
    result = air_quality_tool.get_latest_readings(
        latitude=lat, longitude=lon, radius_meters=radius
    )
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


# ══════════════════════════════════════════════════════════════════
# City Overview Endpoint (all data sources)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/city/overview")
def get_city_overview(
    lat: float = Query(default=51.5074, description="Latitude"),
    lon: float = Query(default=-0.1278, description="Longitude"),
):
    """
    Full city overview: TfL + Weather + Air Quality in one call.
    This is what the agent will use to get a quick snapshot.
    """
    tube = tfl_tool.get_tube_status()
    weather = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    air = air_quality_tool.get_latest_readings(latitude=lat, longitude=lon)

    return {
        "tube": {
            "success": tube.success,
            "summary": tube.summary,
        },
        "weather": {
            "success": weather.success,
            "summary": weather.summary,
        },
        "air_quality": {
            "success": air.success,
            "summary": air.summary,
        },
    }


# ── Run directly ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )

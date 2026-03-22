"""
Smart City AI Agent - FastAPI Application
Day 4: All data source endpoints + LangGraph agent chat endpoint.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.tools.tfl import TfLTool
from app.tools.weather import WeatherTool
from app.tools.air_quality import AirQualityTool
from app.tools.tomtom import TomTomTool
from app.agent.graph import create_agent

# ── Logging Setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global agent instance ─────────────────────────────────────────
agent = None

# ── Lifespan (startup / shutdown) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global agent
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Initialize agent
    if settings.GEMINI_API_KEY:
        agent = create_agent()
        logger.info("✅ LangGraph agent initialized")
    else:
        logger.warning("⚠️ GEMINI_API_KEY not set — agent disabled")

    yield
    logger.info("Shutting down.")


# ── App Instance ──────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="An autonomous AI agent for London city data analysis",
    version="0.4.0",
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
tomtom_tool = TomTomTool()


# ── Request/Response Models ───────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for agent chat."""
    message: str = Field(
        description="User's question about London city conditions",
        min_length=1,
        max_length=1000,
    )


class ChatResponse(BaseModel):
    """Response from the agent."""
    response: str = Field(description="Agent's analysis and answer")
    tools_used: list[str] = Field(description="Tools the agent called")
    success: bool = Field(description="Whether the agent completed successfully")
    error: str | None = Field(default=None, description="Error message if failed")


# ══════════════════════════════════════════════════════════════════
# Agent Chat Endpoint
# ══════════════════════════════════════════════════════════════════

@app.post("/api/agent/chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest):
    """
    Send a question to the Smart City AI Agent.
    The agent will:
    1. Analyze your question
    2. Decide which data sources to query
    3. Fetch real-time data
    4. Correlate and analyze the results
    5. Return an insightful answer
    """
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not available. Set GEMINI_API_KEY in .env",
        )

    logger.info(f"📨 Agent chat: {request.message[:100]}...")

    try:
        # Invoke the LangGraph agent
        result = agent.invoke({
            "messages": [("user", request.message)],
            "tools_to_call": [],
            "tool_results": {},
            "analysis": "",
            "iteration_count": 0,
            "error": "",
        })

        # Extract the final AI message
        messages = result.get("messages", [])
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]

        if ai_messages:
            response_text = ai_messages[-1].content
        else:
            response_text = result.get("analysis", "No response generated.")

        tools_used = result.get("tools_to_call", [])
        error = result.get("error", "")

        return ChatResponse(
            response=response_text,
            tools_used=tools_used,
            success=True,
            error=error if error else None,
        )

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return ChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            tools_used=[],
            success=False,
            error=str(e),
        )


# ── Health Check ──────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "0.4.0",
        "tools_available": ["tfl", "weather", "air_quality", "tomtom"],
        "agent_ready": agent is not None,
    }


# ══════════════════════════════════════════════════════════════════
# TfL Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/tfl/tube-status")
def get_tube_status():
    result = tfl_tool.get_tube_status()
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tfl/disruptions")
def get_disruptions():
    result = tfl_tool.get_road_disruptions()
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tfl/road-status")
def get_road_status(
    road_ids: str | None = Query(default=None),
):
    result = tfl_tool.get_road_status(road_ids=road_ids)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tfl/summary")
def get_tfl_summary():
    tube = tfl_tool.get_tube_status()
    disruptions = tfl_tool.get_road_disruptions()
    return {
        "tube_status": {"success": tube.success, "summary": tube.summary},
        "road_disruptions": {"success": disruptions.success, "summary": disruptions.summary},
    }


# ══════════════════════════════════════════════════════════════════
# Weather Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/weather/current")
def get_current_weather(
    lat: float = Query(default=51.5074),
    lon: float = Query(default=-0.1278),
):
    result = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/weather/forecast")
def get_weather_forecast(
    lat: float = Query(default=51.5074),
    lon: float = Query(default=-0.1278),
    hours: int = Query(default=12, ge=1, le=48),
):
    result = weather_tool.get_forecast(latitude=lat, longitude=lon, hours=hours)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/weather/summary")
def get_weather_summary(
    lat: float = Query(default=51.5074),
    lon: float = Query(default=-0.1278),
):
    current = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    forecast = weather_tool.get_forecast(latitude=lat, longitude=lon, hours=6)
    return {
        "current": {"success": current.success, "summary": current.summary, "data": current.data if current.success else None},
        "forecast_6h": {"success": forecast.success, "summary": forecast.summary},
    }


# ══════════════════════════════════════════════════════════════════
# Air Quality Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/air-quality/stations")
def get_nearby_stations(
    lat: float = Query(default=51.5074),
    lon: float = Query(default=-0.1278),
    radius: int = Query(default=10000),
):
    result = air_quality_tool.get_nearby_stations(latitude=lat, longitude=lon, radius_meters=radius)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/air-quality/latest")
def get_air_quality_latest(
    lat: float = Query(default=51.5074),
    lon: float = Query(default=-0.1278),
    radius: int = Query(default=10000),
):
    result = air_quality_tool.get_latest_readings(latitude=lat, longitude=lon, radius_meters=radius)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


# ══════════════════════════════════════════════════════════════════
# TomTom Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/tomtom/flow")
def get_traffic_flow(
    lat: float = Query(default=51.5074),
    lon: float = Query(default=-0.1278),
    name: str | None = Query(default=None),
):
    result = tomtom_tool.get_traffic_flow(latitude=lat, longitude=lon, location_name=name)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tomtom/multi-flow")
def get_multi_point_flow(
    points: str | None = Query(default=None),
):
    point_list = points.split(",") if points else None
    result = tomtom_tool.get_multi_point_flow(points=point_list)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tomtom/incidents")
def get_traffic_incidents():
    result = tomtom_tool.get_traffic_incidents()
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()


@app.get("/api/tomtom/points")
def get_available_points():
    return TomTomTool.get_available_points()


# ══════════════════════════════════════════════════════════════════
# City Overview
# ══════════════════════════════════════════════════════════════════

@app.get("/api/city/overview")
def get_city_overview(
    lat: float = Query(default=51.5074),
    lon: float = Query(default=-0.1278),
):
    tube = tfl_tool.get_tube_status()
    weather = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    air = air_quality_tool.get_latest_readings(latitude=lat, longitude=lon)
    traffic = tomtom_tool.get_traffic_flow(latitude=lat, longitude=lon)
    return {
        "tube": {"success": tube.success, "summary": tube.summary},
        "weather": {"success": weather.success, "summary": weather.summary},
        "air_quality": {"success": air.success, "summary": air.summary},
        "traffic_flow": {"success": traffic.success, "summary": traffic.summary},
    }


# ── Run directly ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)

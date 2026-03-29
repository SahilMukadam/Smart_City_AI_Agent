"""
Smart City AI Agent - FastAPI Application
Day 8: Structured responses + caching + error handling.
"""

import time
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
from app.agent.graph import create_agent, get_cache
from app.agent.sessions import SessionManager
from app.agent.response_models import AgentResponse, SourceInfo, CorrelationInsight

# ── Logging Setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────
agent = None
session_manager = SessionManager(session_ttl_seconds=1800, max_sessions=100)

# ── Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME}")

    if settings.GEMINI_API_KEY:
        agent = create_agent()
        logger.info("✅ Agent initialized (parallel + caching + correlation)")
    else:
        logger.warning("⚠️ GEMINI_API_KEY not set — agent disabled")

    yield
    logger.info("Shutting down.")


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="An autonomous AI agent for London city data analysis",
    version="0.8.0",
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


# ── Request Model ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None)


# ══════════════════════════════════════════════════════════════════
# Agent Chat Endpoint (structured response)
# ══════════════════════════════════════════════════════════════════

@app.post("/api/agent/chat", response_model=AgentResponse)
def agent_chat(request: ChatRequest):
    """
    Send a question to the Smart City AI Agent.
    Returns structured response with analysis, source metadata,
    correlation insights, timing, and cache stats.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available. Set GEMINI_API_KEY in .env")

    request_start = time.perf_counter()
    logger.info(f"📨 Agent chat: {request.message[:100]}...")

    # Get or create session
    session = session_manager.get_or_create_session(request.session_id)
    session.add_user_message(request.message)

    try:
        history_messages = session.get_recent_messages(max_messages=10)

        result = agent.invoke({
            "messages": history_messages,
            "tools_to_call": [],
            "tool_arguments": {},
            "tool_results": {},
            "source_metadata": [],
            "correlation_insights": "",
            "parsed_insights": [],
            "analysis": "",
            "iteration_count": 0,
            "error": "",
        })

        # Extract response text
        messages = result.get("messages", [])
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        response_text = ai_messages[-1].content if ai_messages else result.get("analysis", "No response generated.")

        # Build source info
        sources = [
            SourceInfo(
                tool_name=m.get("tool_name", "unknown"),
                success=m.get("success", False),
                cached=m.get("cached", False),
                response_time_ms=m.get("response_time_ms", 0),
                error=m.get("error"),
            )
            for m in result.get("source_metadata", [])
        ]

        # Build correlation insights
        insights = [
            CorrelationInsight(
                type=i.get("type", ""),
                title=i.get("title", ""),
                description=i.get("description", ""),
                confidence=i.get("confidence", "medium"),
            )
            for i in result.get("parsed_insights", [])
        ]

        tools_used = result.get("tools_to_call", [])
        error = result.get("error", "")
        total_time = (time.perf_counter() - request_start) * 1000

        # Store in session
        session.add_ai_message(response_text)
        session.add_tools_used(tools_used)

        # Cache stats for this request
        cache = get_cache()
        cache_hits = sum(1 for s in sources if s.cached)
        cache_misses = sum(1 for s in sources if not s.cached and s.success)

        return AgentResponse(
            response=response_text,
            success=True,
            session_id=session.session_id,
            tools_used=tools_used,
            sources=sources,
            insights=insights,
            total_time_ms=round(total_time, 1),
            error=error if error else None,
            cache_stats={
                "hits_this_request": cache_hits,
                "misses_this_request": cache_misses,
                "global": cache.get_stats(),
            },
        )

    except Exception as e:
        logger.error(f"Agent error: {e}")
        total_time = (time.perf_counter() - request_start) * 1000
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        session.add_ai_message(error_msg)

        return AgentResponse(
            response=error_msg,
            success=False,
            session_id=session.session_id,
            total_time_ms=round(total_time, 1),
            error=str(e),
        )


# ══════════════════════════════════════════════════════════════════
# Session Endpoints
# ══════════════════════════════════════════════════════════════════

@app.post("/api/sessions")
def create_session():
    session = session_manager.create_session()
    return session.get_summary()


@app.get("/api/sessions")
def list_sessions():
    return session_manager.list_sessions()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session.get_summary()


@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return {
        "session_id": session_id,
        "messages": [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in session.messages
        ],
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Session {session_id} deleted"}


# ══════════════════════════════════════════════════════════════════
# Cache Management Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/cache/stats")
def get_cache_stats():
    """Get cache performance statistics."""
    return get_cache().get_stats()


@app.delete("/api/cache")
def clear_cache():
    """Clear the response cache."""
    get_cache().invalidate()
    return {"message": "Cache cleared"}


# ── Health Check ──────────────────────────────────────────────────
@app.get("/health")
def health_check():
    cache = get_cache()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "0.8.0",
        "tools_available": ["tfl", "weather", "air_quality", "tomtom"],
        "agent_ready": agent is not None,
        "active_sessions": session_manager.active_count,
        "cache_stats": cache.get_stats(),
        "features": [
            "parallel_execution", "conditional_routing", "argument_extraction",
            "conversation_memory", "correlation_engine", "response_caching",
            "structured_output",
        ],
    }


# ══════════════════════════════════════════════════════════════════
# Data Source Endpoints (unchanged)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/tfl/tube-status")
def get_tube_status():
    result = tfl_tool.get_tube_status()
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/tfl/disruptions")
def get_disruptions():
    result = tfl_tool.get_road_disruptions()
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/tfl/road-status")
def get_road_status(road_ids: str | None = Query(default=None)):
    result = tfl_tool.get_road_status(road_ids=road_ids)
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/tfl/summary")
def get_tfl_summary():
    tube = tfl_tool.get_tube_status()
    disruptions = tfl_tool.get_road_disruptions()
    return {
        "tube_status": {"success": tube.success, "summary": tube.summary},
        "road_disruptions": {"success": disruptions.success, "summary": disruptions.summary},
    }

@app.get("/api/weather/current")
def get_current_weather(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278)):
    result = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/weather/forecast")
def get_weather_forecast(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278), hours: int = Query(default=12, ge=1, le=48)):
    result = weather_tool.get_forecast(latitude=lat, longitude=lon, hours=hours)
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/weather/summary")
def get_weather_summary(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278)):
    current = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    forecast = weather_tool.get_forecast(latitude=lat, longitude=lon, hours=6)
    return {
        "current": {"success": current.success, "summary": current.summary, "data": current.data if current.success else None},
        "forecast_6h": {"success": forecast.success, "summary": forecast.summary},
    }

@app.get("/api/air-quality/stations")
def get_nearby_stations(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278), radius: int = Query(default=10000)):
    result = air_quality_tool.get_nearby_stations(latitude=lat, longitude=lon, radius_meters=radius)
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/air-quality/latest")
def get_air_quality_latest(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278), radius: int = Query(default=10000)):
    result = air_quality_tool.get_latest_readings(latitude=lat, longitude=lon, radius_meters=radius)
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/tomtom/flow")
def get_traffic_flow(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278), name: str | None = Query(default=None)):
    result = tomtom_tool.get_traffic_flow(latitude=lat, longitude=lon, location_name=name)
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/tomtom/multi-flow")
def get_multi_point_flow(points: str | None = Query(default=None)):
    point_list = points.split(",") if points else None
    result = tomtom_tool.get_multi_point_flow(points=point_list)
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/tomtom/incidents")
def get_traffic_incidents():
    result = tomtom_tool.get_traffic_incidents()
    if not result.success: raise HTTPException(status_code=502, detail=result.error)
    return result.model_dump()

@app.get("/api/tomtom/points")
def get_available_points():
    return TomTomTool.get_available_points()

@app.get("/api/city/overview")
def get_city_overview(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278)):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)

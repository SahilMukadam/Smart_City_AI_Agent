"""
Smart City AI Agent - FastAPI Application
Day 9: Anomaly detection + city health score + proactive insights.
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
from app.agent.response_models import (
    AgentResponse, SourceInfo, CorrelationInsight,
    AnomalyAlert, HealthScores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

agent = None
session_manager = SessionManager(session_ttl_seconds=1800, max_sessions=100)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME}")
    if settings.GEMINI_API_KEY:
        agent = create_agent()
        logger.info("✅ Agent initialized (full intelligence layer)")
    else:
        logger.warning("⚠️ GEMINI_API_KEY not set — agent disabled")
    yield
    logger.info("Shutting down.")


settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    description="An autonomous AI agent for London city data analysis",
    version="0.9.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

tfl_tool = TfLTool()
weather_tool = WeatherTool()
air_quality_tool = AirQualityTool()
tomtom_tool = TomTomTool()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None)


# ══════════════════════════════════════════════════════════════════
# Agent Chat Endpoint
# ══════════════════════════════════════════════════════════════════

@app.post("/api/agent/chat", response_model=AgentResponse)
def agent_chat(request: ChatRequest):
    """Send a question to the Smart City AI Agent."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available. Set GEMINI_API_KEY in .env")

    request_start = time.perf_counter()
    session = session_manager.get_or_create_session(request.session_id)
    session.add_user_message(request.message)

    try:
        result = agent.invoke({
            "messages": session.get_recent_messages(max_messages=10),
            "tools_to_call": [], "tool_arguments": {}, "tool_results": {},
            "source_metadata": [],
            "correlation_insights": "", "parsed_insights": [],
            "anomaly_alerts": "", "parsed_anomalies": [], "health_scores": {},
            "analysis": "", "iteration_count": 0, "error": "",
        })

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        response_text = ai_messages[-1].content if ai_messages else result.get("analysis", "No response.")

        sources = [SourceInfo(**m) for m in result.get("source_metadata", [])]
        insights = [CorrelationInsight(**i) for i in result.get("parsed_insights", [])]
        anomalies = [AnomalyAlert(**a) for a in result.get("parsed_anomalies", [])]

        raw_health = result.get("health_scores", {})
        health = HealthScores(**raw_health) if raw_health else None

        total_time = (time.perf_counter() - request_start) * 1000
        session.add_ai_message(response_text)
        session.add_tools_used(result.get("tools_to_call", []))

        cache = get_cache()
        cache_hits = sum(1 for s in sources if s.cached)

        return AgentResponse(
            response=response_text, success=True, session_id=session.session_id,
            tools_used=result.get("tools_to_call", []),
            sources=sources, insights=insights, anomalies=anomalies, health=health,
            total_time_ms=round(total_time, 1),
            error=result.get("error") or None,
            cache_stats={"hits": cache_hits, "global": cache.get_stats()},
        )

    except Exception as e:
        logger.error(f"Agent error: {e}")
        total_time = (time.perf_counter() - request_start) * 1000
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        session.add_ai_message(error_msg)
        return AgentResponse(
            response=error_msg, success=False, session_id=session.session_id,
            total_time_ms=round(total_time, 1), error=str(e),
        )


# ══════════════════════════════════════════════════════════════════
# Proactive City Insights Endpoint
# ══════════════════════════════════════════════════════════════════

@app.get("/api/city/insights")
def get_city_insights():
    """
    Proactive city insights — runs all data sources without a user question
    and returns anomalies, correlations, and health scores.
    
    Uses ~2 Gemini API calls (router + analyzer).
    Great for dashboards that want a periodic health check.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    request_start = time.perf_counter()

    try:
        result = agent.invoke({
            "messages": [("user", "Give me a complete London city overview with all data sources. Check traffic, tube, weather, and air quality.")],
            "tools_to_call": [], "tool_arguments": {}, "tool_results": {},
            "source_metadata": [],
            "correlation_insights": "", "parsed_insights": [],
            "anomaly_alerts": "", "parsed_anomalies": [], "health_scores": {},
            "analysis": "", "iteration_count": 0, "error": "",
        })

        messages = result.get("messages", [])
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        summary = ai_messages[-1].content if ai_messages else result.get("analysis", "")

        total_time = (time.perf_counter() - request_start) * 1000

        return {
            "summary": summary,
            "health_scores": result.get("health_scores", {}),
            "anomalies": result.get("parsed_anomalies", []),
            "insights": result.get("parsed_insights", []),
            "sources": result.get("source_metadata", []),
            "tools_used": result.get("tools_to_call", []),
            "total_time_ms": round(total_time, 1),
        }

    except Exception as e:
        logger.error(f"Insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════
# Session Endpoints
# ══════════════════════════════════════════════════════════════════

@app.post("/api/sessions")
def create_session():
    return session_manager.create_session().get_summary()

@app.get("/api/sessions")
def list_sessions():
    return session_manager.list_sessions()

@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    return session.get_summary()

@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str):
    session = session_manager.get_session(session_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in session.messages
    ]}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Session {session_id} deleted"}


# ══════════════════════════════════════════════════════════════════
# Cache Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/cache/stats")
def get_cache_stats():
    return get_cache().get_stats()

@app.delete("/api/cache")
def clear_cache():
    get_cache().invalidate()
    return {"message": "Cache cleared"}


# ══════════════════════════════════════════════════════════════════
# Health Check
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    return {
        "status": "healthy", "app": settings.APP_NAME, "version": "0.9.0",
        "tools_available": ["tfl", "weather", "air_quality", "tomtom"],
        "agent_ready": agent is not None,
        "active_sessions": session_manager.active_count,
        "cache_stats": get_cache().get_stats(),
        "features": [
            "parallel_execution", "conditional_routing", "argument_extraction",
            "conversation_memory", "correlation_engine", "anomaly_detection",
            "city_health_score", "response_caching", "structured_output",
            "proactive_insights",
        ],
    }


# ══════════════════════════════════════════════════════════════════
# Data Source Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/tfl/tube-status")
def get_tube_status():
    r = tfl_tool.get_tube_status()
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/tfl/disruptions")
def get_disruptions():
    r = tfl_tool.get_road_disruptions()
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/tfl/road-status")
def get_road_status(road_ids: str | None = Query(default=None)):
    r = tfl_tool.get_road_status(road_ids=road_ids)
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/tfl/summary")
def get_tfl_summary():
    t = tfl_tool.get_tube_status(); d = tfl_tool.get_road_disruptions()
    return {"tube_status": {"success": t.success, "summary": t.summary}, "road_disruptions": {"success": d.success, "summary": d.summary}}

@app.get("/api/weather/current")
def get_current_weather(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278)):
    r = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/weather/forecast")
def get_weather_forecast(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278), hours: int = Query(default=12, ge=1, le=48)):
    r = weather_tool.get_forecast(latitude=lat, longitude=lon, hours=hours)
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/weather/summary")
def get_weather_summary(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278)):
    c = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    f = weather_tool.get_forecast(latitude=lat, longitude=lon, hours=6)
    return {"current": {"success": c.success, "summary": c.summary, "data": c.data if c.success else None}, "forecast_6h": {"success": f.success, "summary": f.summary}}

@app.get("/api/air-quality/stations")
def get_nearby_stations(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278), radius: int = Query(default=10000)):
    r = air_quality_tool.get_nearby_stations(latitude=lat, longitude=lon, radius_meters=radius)
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/air-quality/latest")
def get_air_quality_latest(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278), radius: int = Query(default=10000)):
    r = air_quality_tool.get_latest_readings(latitude=lat, longitude=lon, radius_meters=radius)
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/tomtom/flow")
def get_traffic_flow(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278), name: str | None = Query(default=None)):
    r = tomtom_tool.get_traffic_flow(latitude=lat, longitude=lon, location_name=name)
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/tomtom/multi-flow")
def get_multi_point_flow(points: str | None = Query(default=None)):
    r = tomtom_tool.get_multi_point_flow(points=points.split(",") if points else None)
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/tomtom/incidents")
def get_traffic_incidents():
    r = tomtom_tool.get_traffic_incidents()
    if not r.success: raise HTTPException(status_code=502, detail=r.error)
    return r.model_dump()

@app.get("/api/tomtom/points")
def get_available_points():
    return TomTomTool.get_available_points()

@app.get("/api/city/overview")
def get_city_overview(lat: float = Query(default=51.5074), lon: float = Query(default=-0.1278)):
    t = tfl_tool.get_tube_status(); w = weather_tool.get_current_weather(latitude=lat, longitude=lon)
    a = air_quality_tool.get_latest_readings(latitude=lat, longitude=lon)
    tr = tomtom_tool.get_traffic_flow(latitude=lat, longitude=lon)
    return {"tube": {"success": t.success, "summary": t.summary}, "weather": {"success": w.success, "summary": w.summary},
            "air_quality": {"success": a.success, "summary": a.summary}, "traffic_flow": {"success": tr.success, "summary": tr.summary}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)

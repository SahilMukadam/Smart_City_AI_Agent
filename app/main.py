"""
Smart City AI Agent - FastAPI Application
Day 11: Reasoning chain + example query library.
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
from app.agent.reasoning import build_reasoning_steps, EXAMPLE_QUERIES
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
        logger.info("✅ Agent initialized")
    else:
        logger.warning("⚠️ GEMINI_API_KEY not set")
    yield
    logger.info("Shutting down.")


settings = get_settings()
app = FastAPI(title=settings.APP_NAME, version="0.11.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

tfl_tool = TfLTool()
weather_tool = WeatherTool()
air_quality_tool = AirQualityTool()
tomtom_tool = TomTomTool()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None)


# ══════════════════════════════════════════════════════════════════
# Agent Chat Endpoint (with reasoning steps)
# ══════════════════════════════════════════════════════════════════

@app.post("/api/agent/chat")
def agent_chat(request: ChatRequest):
    """Send a question to the agent. Returns structured response with reasoning steps."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    request_start = time.perf_counter()
    session = session_manager.get_or_create_session(request.session_id)
    session.add_user_message(request.message)

    try:
        history_messages = session.get_recent_messages(max_messages=10)

        initial_state = {
            "messages": history_messages,
            "tools_to_call": [], "tool_arguments": {}, "tool_results": {},
            "source_metadata": [],
            "correlation_insights": "", "parsed_insights": [],
            "anomaly_alerts": "", "parsed_anomalies": [], "health_scores": {},
            "analysis": "", "iteration_count": 0, "error": "",
        }

        # Run with step-by-step streaming to capture node timings
        node_order = []
        node_timings = {}
        current_state = initial_state

        for step_output in agent.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in step_output.items():
                step_start = time.perf_counter()
                node_order.append(node_name)
                # Merge output into current state for reasoning detail extraction
                for key, value in node_output.items():
                    current_state[key] = value
                node_timings[node_name] = (time.perf_counter() - step_start) * 1000

        # Build the final result from accumulated state
        result = current_state

        # Extract response
        messages = result.get("messages", [])
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        response_text = ai_messages[-1].content if ai_messages else result.get("analysis", "No response.")

        # Build reasoning steps
        reasoning_steps = build_reasoning_steps(result, node_order, node_timings)

        sources = [SourceInfo(**m) for m in result.get("source_metadata", [])]
        insights = [CorrelationInsight(**i) for i in result.get("parsed_insights", [])]
        anomalies = [AnomalyAlert(**a) for a in result.get("parsed_anomalies", [])]
        raw_health = result.get("health_scores", {})
        health = HealthScores(**raw_health) if raw_health else None

        total_time = (time.perf_counter() - request_start) * 1000
        session.add_ai_message(response_text)
        session.add_tools_used(result.get("tools_to_call", []))

        cache = get_cache()

        return {
            "response": response_text,
            "success": True,
            "session_id": session.session_id,
            "tools_used": result.get("tools_to_call", []),
            "sources": [s.model_dump() for s in sources],
            "insights": [i.model_dump() for i in insights],
            "anomalies": [a.model_dump() for a in anomalies],
            "health": health.model_dump() if health else None,
            "reasoning_steps": reasoning_steps,
            "total_time_ms": round(total_time, 1),
            "error": result.get("error") or None,
            "cache_stats": {"global": cache.get_stats()},
        }

    except Exception as e:
        logger.error(f"Agent error: {e}")
        total_time = (time.perf_counter() - request_start) * 1000
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        session.add_ai_message(error_msg)
        return {
            "response": error_msg, "success": False,
            "session_id": session.session_id,
            "tools_used": [], "sources": [], "insights": [],
            "anomalies": [], "health": None, "reasoning_steps": [],
            "total_time_ms": round(total_time, 1), "error": str(e),
            "cache_stats": {},
        }


# ══════════════════════════════════════════════════════════════════
# Example Queries Endpoint
# ══════════════════════════════════════════════════════════════════

@app.get("/api/examples")
def get_example_queries():
    """Get curated example queries for the demo library."""
    return EXAMPLE_QUERIES


# ══════════════════════════════════════════════════════════════════
# Proactive Insights
# ══════════════════════════════════════════════════════════════════

@app.get("/api/city/insights")
def get_city_insights():
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    request_start = time.perf_counter()
    try:
        initial_state = {
            "messages": [("user", "Full London city overview — traffic, tube, weather, and air quality")],
            "tools_to_call": [], "tool_arguments": {}, "tool_results": {},
            "source_metadata": [],
            "correlation_insights": "", "parsed_insights": [],
            "anomaly_alerts": "", "parsed_anomalies": [], "health_scores": {},
            "analysis": "", "iteration_count": 0, "error": "",
        }

        current_state = initial_state
        for step_output in agent.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in step_output.items():
                for key, value in node_output.items():
                    current_state[key] = value

        messages = current_state.get("messages", [])
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        summary = ai_messages[-1].content if ai_messages else current_state.get("analysis", "")

        return {
            "summary": summary,
            "health_scores": current_state.get("health_scores", {}),
            "anomalies": current_state.get("parsed_anomalies", []),
            "insights": current_state.get("parsed_insights", []),
            "sources": current_state.get("source_metadata", []),
            "tools_used": current_state.get("tools_to_call", []),
            "total_time_ms": round((time.perf_counter() - request_start) * 1000, 1),
        }
    except Exception as e:
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
    s = session_manager.get_session(session_id)
    if not s: raise HTTPException(status_code=404, detail="Session not found")
    return s.get_summary()

@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str):
    s = session_manager.get_session(session_id)
    if not s: raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in s.messages]}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not session_manager.delete_session(session_id): raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Session {session_id} deleted"}

# ── Cache ─────────────────────────────────────────────────────────
@app.get("/api/cache/stats")
def get_cache_stats():
    return get_cache().get_stats()

@app.delete("/api/cache")
def clear_cache():
    get_cache().invalidate()
    return {"message": "Cache cleared"}

# ── Health ────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "healthy", "app": settings.APP_NAME, "version": "0.11.0",
        "tools_available": ["tfl", "weather", "air_quality", "tomtom"],
        "agent_ready": agent is not None,
        "active_sessions": session_manager.active_count,
        "cache_stats": get_cache().get_stats(),
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
    a = air_quality_tool.get_latest_readings(latitude=lat, longitude=lon); tr = tomtom_tool.get_traffic_flow(latitude=lat, longitude=lon)
    return {"tube": {"success": t.success, "summary": t.summary}, "weather": {"success": w.success, "summary": w.summary},
            "air_quality": {"success": a.success, "summary": a.summary}, "traffic_flow": {"success": tr.success, "summary": tr.summary}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)

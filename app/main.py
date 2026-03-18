"""
Smart City AI Agent - FastAPI Application
Day 1: Health check + TfL diagnostic endpoints.
More endpoints added as tools are built.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.tools.tfl import TfLTool

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
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Tool Instances ────────────────────────────────────────────────
tfl_tool = TfLTool()


# ── Health Check ──────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "0.1.0",
        "tools_available": ["tfl"],
    }


# ── TfL Endpoints (diagnostic / testing) ─────────────────────────

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
    """
    Quick summary of all TfL data - tube + disruptions + roads.
    Useful for the agent to get a full picture in one call.
    """
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


# ── Run directly ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )

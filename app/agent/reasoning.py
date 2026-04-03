"""
Smart City AI Agent - Reasoning Tracker
Captures step-by-step node transitions during agent execution.
Returns structured steps for the frontend to display.
"""

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning chain."""
    node: str
    label: str
    emoji: str
    detail: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "label": self.label,
            "emoji": self.emoji,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 1),
        }


# ── Node display config ──────────────────────────────────────────
NODE_CONFIG = {
    "router": {
        "emoji": "🔀",
        "label": "Selecting tools",
        "detail_fn": lambda state: (
            f"Selected: {', '.join(state.get('tools_to_call', []))}"
            if state.get("tools_to_call")
            else "No tools needed — direct response"
        ),
    },
    "argument_extractor": {
        "emoji": "🎯",
        "label": "Extracting parameters",
        "detail_fn": lambda state: _format_args_detail(state.get("tool_arguments", {})),
    },
    "tool_executor": {
        "emoji": "📡",
        "label": "Fetching live data",
        "detail_fn": lambda state: _format_executor_detail(state.get("source_metadata", [])),
    },
    "correlator": {
        "emoji": "📊",
        "label": "Analyzing patterns",
        "detail_fn": lambda state: _format_correlator_detail(state),
    },
    "analyzer": {
        "emoji": "🧠",
        "label": "Generating analysis",
        "detail_fn": lambda state: "Correlating data with Gemini...",
    },
    "responder": {
        "emoji": "💬",
        "label": "Formatting response",
        "detail_fn": lambda state: _format_responder_detail(state),
    },
    "direct_responder": {
        "emoji": "💬",
        "label": "Responding directly",
        "detail_fn": lambda state: "No data tools needed for this query",
    },
}


def _format_args_detail(tool_args: dict) -> str:
    if not tool_args:
        return "Using default London coordinates"
    parts = []
    for tool_key, args in tool_args.items():
        name = tool_key.split("__")[0].replace("get_", "").replace("_", " ").title()
        if args:
            loc = args.get("location_name", "")
            if loc:
                parts.append(f"{name} → {loc}")
            else:
                lat = args.get("latitude", "")
                if lat:
                    parts.append(f"{name} → ({lat:.2f}, {args.get('longitude', 0):.2f})")
                else:
                    parts.append(name)
        else:
            parts.append(f"{name} (defaults)")
    return ", ".join(parts) if parts else "Using defaults"


def _format_executor_detail(metadata: list[dict]) -> str:
    if not metadata:
        return "No tools executed"
    cached = sum(1 for m in metadata if m.get("cached"))
    ok = sum(1 for m in metadata if m.get("success") and not m.get("cached"))
    failed = sum(1 for m in metadata if not m.get("success"))
    parts = []
    if ok:
        parts.append(f"{ok} API call(s)")
    if cached:
        parts.append(f"{cached} cached")
    if failed:
        parts.append(f"{failed} failed")
    return ", ".join(parts)


def _format_correlator_detail(state: dict) -> str:
    insights = state.get("parsed_insights", [])
    anomalies = state.get("parsed_anomalies", [])
    health = state.get("health_scores", {})
    overall = health.get("overall")

    parts = []
    if insights:
        parts.append(f"{len(insights)} correlation(s)")
    if anomalies:
        critical = sum(1 for a in anomalies if a.get("level") == "critical")
        warning = sum(1 for a in anomalies if a.get("level") == "warning")
        if critical:
            parts.append(f"{critical} critical alert(s)")
        if warning:
            parts.append(f"{warning} warning(s)")
    if overall is not None:
        parts.append(f"Health: {overall}/100")
    return ", ".join(parts) if parts else "Analyzing..."


def _format_responder_detail(state: dict) -> str:
    tool_results = state.get("tool_results", {})
    successful = sum(1 for v in tool_results.values() if not v.startswith("ERROR"))
    return f"Compiled from {successful} data source(s)"


def build_reasoning_steps(result: dict, node_order: list[str], node_timings: dict[str, float]) -> list[dict]:
    """
    Build the reasoning steps from a completed agent run.
    Args:
        result: The final agent state after invoke()
        node_order: List of nodes that were visited
        node_timings: Dict of node_name → duration_ms
    """
    steps = []
    for node_name in node_order:
        config = NODE_CONFIG.get(node_name)
        if not config:
            continue

        detail = ""
        try:
            detail = config["detail_fn"](result)
        except Exception:
            detail = ""

        steps.append(ReasoningStep(
            node=node_name,
            label=config["label"],
            emoji=config["emoji"],
            detail=detail,
            duration_ms=node_timings.get(node_name, 0),
        ).to_dict())

    return steps


# ── Example Queries Library ───────────────────────────────────────

EXAMPLE_QUERIES = [
    {
        "category": "🚗 Traffic",
        "queries": [
            {"text": "How's the traffic in Central London right now?", "description": "Single-point traffic check"},
            {"text": "Give me a London-wide traffic overview", "description": "Multi-point congestion scan"},
            {"text": "Why is traffic bad? Check weather and incidents too", "description": "Multi-source correlation"},
            {"text": "Compare traffic between Canary Wharf and City of London", "description": "Location comparison"},
        ],
    },
    {
        "category": "🚇 Transport",
        "queries": [
            {"text": "How's the tube today?", "description": "Tube line status check"},
            {"text": "Are there any road disruptions in London?", "description": "TfL road disruptions"},
            {"text": "Tube status and traffic — is it better to drive or take the underground?", "description": "Multi-source decision support"},
        ],
    },
    {
        "category": "🌤️ Weather",
        "queries": [
            {"text": "What's the weather like in London?", "description": "Current conditions"},
            {"text": "Will it rain in the next 6 hours?", "description": "Short-term forecast"},
            {"text": "How is the weather affecting traffic today?", "description": "Weather-traffic correlation"},
        ],
    },
    {
        "category": "💨 Air Quality",
        "queries": [
            {"text": "How's the air quality in Central London?", "description": "PM2.5, NO₂ readings + AQI"},
            {"text": "Is air pollution related to traffic congestion right now?", "description": "Traffic-AQ correlation"},
            {"text": "Is it safe to go for a run outside today?", "description": "Health recommendation"},
        ],
    },
    {
        "category": "🏙️ Full Reports",
        "queries": [
            {"text": "Full London city overview — traffic, tube, weather, and air quality", "description": "All data sources + correlations"},
            {"text": "What's happening in London right now? Any anomalies?", "description": "Anomaly-focused report"},
            {"text": "How's Camden doing? Check everything", "description": "Location-specific full report"},
        ],
    },
]

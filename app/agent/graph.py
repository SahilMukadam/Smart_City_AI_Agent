"""
Smart City AI Agent - LangGraph Agent
The core agent graph: router → tool_executor → analyzer → responder.

Day 4: Linear flow (sequential tool calls).
Day 5: Parallel tool execution will be added.
"""

import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from app.config import get_settings
from app.agent.state import CityAgentState
from app.agent.tools import ALL_TOOLS, TOOL_MAP

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
MAX_ITERATIONS = 3  # Safety limit to prevent infinite loops

SYSTEM_PROMPT = """You are the Smart City AI Agent, an expert analyst for London city conditions.
You have access to real-time data tools for:
- Transport for London (TfL): tube/underground status, road disruptions, road corridor conditions
- Weather: current conditions and forecasts (Open-Meteo)
- Air Quality: pollution readings from monitoring stations (OpenAQ)
- Traffic Flow: real-time speed and congestion data (TomTom)

Your job:
1. Understand the user's question about London city conditions
2. Decide which data tools to call to answer it
3. Analyze the data you receive, looking for correlations and patterns
4. Give a clear, insightful answer with specific data points

Guidelines:
- Always cite specific numbers (speeds, temperatures, AQI values)
- Look for correlations: e.g., rain + rush hour → worse congestion
- If data from one source is unavailable, work with what you have
- Be concise but thorough. Lead with the key insight, then supporting data.
- When comparing areas, present data in a structured way
- Current date/time: {current_time}
"""

ROUTER_PROMPT = """Based on the user's question, decide which tools to call.

Available tools:
- get_tube_status: London Underground line status and delays
- get_road_disruptions: TfL road disruptions, roadworks, closures
- get_road_corridor_status: Status of specific major roads (A1, A2, etc.)
- get_current_weather: Current temperature, rain, wind, humidity
- get_weather_forecast: Hourly forecast for next 1-48 hours
- get_air_quality: PM2.5, PM10, NO2 readings with AQI category
- get_traffic_flow: Real-time speed/congestion at a specific point
- get_london_traffic_overview: Traffic across 10 key London locations
- get_traffic_incidents: Accidents, roadworks, jams from TomTom

Return ONLY a comma-separated list of tool names to call. Choose the minimum set needed.

Examples:
- "How's the tube?" → get_tube_status
- "Why is traffic bad?" → get_traffic_flow,get_road_disruptions,get_current_weather
- "What's it like outside?" → get_current_weather,get_air_quality
- "London overview" → get_tube_status,get_current_weather,get_london_traffic_overview
- "Will it rain?" → get_weather_forecast
- "Air quality and traffic in Central London" → get_air_quality,get_traffic_flow

User question: {question}

Tools to call (comma-separated, no spaces):"""

ANALYSIS_PROMPT = """You are analyzing real-time London city data to answer the user's question.

User question: {question}

Data collected from tools:
{tool_data}

Provide a clear, insightful analysis that:
1. Directly answers the user's question
2. Cites specific numbers and data points
3. Identifies any correlations between data sources (e.g., weather affecting traffic)
4. Notes anything unusual or noteworthy
5. Is concise — lead with the key insight, then supporting details

If some data sources failed or returned no data, work with what's available and note the gap briefly."""


# ══════════════════════════════════════════════════════════════════
# LLM Instance
# ══════════════════════════════════════════════════════════════════

def _get_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini LLM instance."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
        max_output_tokens=1024,
    )


# ══════════════════════════════════════════════════════════════════
# Graph Nodes
# ══════════════════════════════════════════════════════════════════

def router_node(state: CityAgentState) -> dict:
    """
    NODE 1: Router
    Reads the user's question, decides which tools to call.
    Uses Gemini to classify the query and select appropriate tools.
    """
    logger.info("🔀 Router: Deciding which tools to call...")

    # Get the latest user message
    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    try:
        llm = _get_llm()
        prompt = ROUTER_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])

        # Parse the comma-separated tool names
        raw_tools = response.content.strip()
        # Clean up: remove any backticks, quotes, or extra whitespace
        raw_tools = raw_tools.replace("`", "").replace('"', "").replace("'", "")
        tool_names = [t.strip() for t in raw_tools.split(",") if t.strip()]

        # Validate tool names — only keep ones that actually exist
        valid_tools = [t for t in tool_names if t in TOOL_MAP]

        if not valid_tools:
            # Fallback: if LLM returned garbage, use a sensible default
            logger.warning(f"Router returned no valid tools from: {raw_tools}")
            valid_tools = ["get_tube_status", "get_current_weather"]

        logger.info(f"🔀 Router selected tools: {valid_tools}")

        return {
            "tools_to_call": valid_tools,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    except Exception as e:
        logger.error(f"Router error: {e}")
        return {
            "tools_to_call": ["get_tube_status", "get_current_weather"],
            "error": f"Router error: {str(e)}",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }


def tool_executor_node(state: CityAgentState) -> dict:
    """
    NODE 2: Tool Executor
    Calls each selected tool sequentially and collects results.
    Day 5 will upgrade this to parallel execution.
    """
    tools_to_call = state.get("tools_to_call", [])
    logger.info(f"🔧 Tool Executor: Calling {len(tools_to_call)} tools...")

    results: dict[str, str] = {}

    for tool_name in tools_to_call:
        tool_func = TOOL_MAP.get(tool_name)
        if not tool_func:
            results[tool_name] = f"ERROR: Unknown tool '{tool_name}'"
            continue

        try:
            logger.info(f"  📡 Calling {tool_name}...")
            # Invoke the tool with no arguments (defaults to London)
            result = tool_func.invoke({})
            results[tool_name] = result
            logger.info(f"  ✅ {tool_name} returned data")

        except Exception as e:
            logger.error(f"  ❌ {tool_name} failed: {e}")
            results[tool_name] = f"ERROR: {str(e)}"

    return {"tool_results": results}


def analyzer_node(state: CityAgentState) -> dict:
    """
    NODE 3: Analyzer
    Takes all tool results, sends them to Gemini for correlation
    and insight generation.
    """
    logger.info("🧠 Analyzer: Correlating data...")

    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    tool_results = state.get("tool_results", {})

    # Format tool results for the LLM
    tool_data_parts = []
    for tool_name, result in tool_results.items():
        tool_data_parts.append(f"--- {tool_name} ---\n{result}")
    tool_data = "\n\n".join(tool_data_parts)

    try:
        llm = _get_llm()
        current_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        system = SYSTEM_PROMPT.format(current_time=current_time)
        analysis_prompt = ANALYSIS_PROMPT.format(
            question=question,
            tool_data=tool_data,
        )

        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=analysis_prompt),
        ])

        analysis = response.content.strip()
        logger.info("🧠 Analyzer: Analysis complete")

        return {"analysis": analysis}

    except Exception as e:
        logger.error(f"Analyzer error: {e}")
        # Fallback: return raw tool summaries if LLM fails
        fallback = "I collected the following data but couldn't generate a full analysis:\n\n"
        fallback += tool_data
        return {
            "analysis": fallback,
            "error": f"Analysis error: {str(e)}",
        }


def responder_node(state: CityAgentState) -> dict:
    """
    NODE 4: Responder
    Formats the analysis as the final AI message in the conversation.
    """
    logger.info("💬 Responder: Formatting final answer...")

    analysis = state.get("analysis", "No analysis available.")
    tools_used = state.get("tools_to_call", [])
    tool_results = state.get("tool_results", {})

    # Build the response with metadata
    response_parts = [analysis]

    # Add data sources footer
    successful_tools = [t for t, r in tool_results.items() if not r.startswith("ERROR")]
    if successful_tools:
        sources = ", ".join(successful_tools)
        response_parts.append(f"\n\n📊 *Data sources: {sources}*")

    final_response = "\n".join(response_parts)

    return {
        "messages": [AIMessage(content=final_response)],
    }


# ══════════════════════════════════════════════════════════════════
# Graph Builder
# ══════════════════════════════════════════════════════════════════

def build_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent.

    Flow: START → router → tool_executor → analyzer → responder → END

    Returns a compiled graph that can be invoked with:
        result = agent.invoke({"messages": [("user", "question")]})
    """
    graph = StateGraph(CityAgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("responder", responder_node)

    # Add edges (linear flow for Day 4)
    graph.add_edge(START, "router")
    graph.add_edge("router", "tool_executor")
    graph.add_edge("tool_executor", "analyzer")
    graph.add_edge("analyzer", "responder")
    graph.add_edge("responder", END)

    # Compile
    agent = graph.compile()
    logger.info("✅ Agent graph compiled successfully")

    return agent


# ── Convenience function ──────────────────────────────────────────

def create_agent():
    """Create and return a ready-to-use agent instance."""
    return build_agent_graph()

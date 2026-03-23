"""
Smart City AI Agent - LangGraph Agent (Day 5)
Parallel tool execution + conditional routing + argument extraction.

Graph flow:
  START → router → [should_use_tools?]
                      ├── yes → tool_executor (parallel) → analyzer → responder → END
                      └── no  → direct_responder → END
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from app.config import get_settings
from app.agent.state import CityAgentState
from app.agent.tools import ALL_TOOLS, TOOL_MAP

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
MAX_ITERATIONS = 3
MAX_PARALLEL_WORKERS = 4  # Max concurrent API calls

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

RULES:
- If the user is just greeting you (hi, hello, hey) or asking a non-city question, respond with: NONE
- Otherwise, return a comma-separated list of tool names. Choose the minimum set needed.
- For broad questions ("how's London?"), use: get_tube_status,get_current_weather,get_london_traffic_overview

Examples:
- "Hi there" → NONE
- "How's the tube?" → get_tube_status
- "Why is traffic bad?" → get_traffic_flow,get_road_disruptions,get_current_weather
- "What's it like outside?" → get_current_weather,get_air_quality
- "London overview" → get_tube_status,get_current_weather,get_london_traffic_overview
- "Will it rain?" → get_weather_forecast
- "Compare traffic in Canary Wharf vs City" → get_traffic_flow,get_traffic_flow
- "Air quality and traffic in Central London" → get_air_quality,get_traffic_flow

User question: {question}

Tools to call (comma-separated list, or NONE):"""

ARGUMENT_PROMPT = """You are deciding what arguments to pass to city data tools.

User question: {question}
Tools to call: {tools}

For each tool, decide what arguments to pass. Most tools default to Central London (51.5074, -0.1278) which is fine for general queries.

Available tool arguments:
- get_traffic_flow: latitude (float), longitude (float), location_name (str)
- get_current_weather: latitude (float), longitude (float)
- get_weather_forecast: latitude (float), longitude (float), hours (int 1-48)
- get_air_quality: latitude (float), longitude (float)
- get_road_corridor_status: road_ids (str, comma-separated like "A1,A2")
- get_tube_status: (no arguments)
- get_road_disruptions: (no arguments)
- get_london_traffic_overview: (no arguments)
- get_traffic_incidents: (no arguments)

Known London locations:
- Central London: 51.5074, -0.1278
- City of London: 51.5155, -0.0922
- Westminster: 51.4975, -0.1357
- Camden: 51.5390, -0.1426
- Tower Bridge: 51.5055, -0.0754
- King's Cross: 51.5317, -0.1240
- Canary Wharf: 51.5054, -0.0235
- Shoreditch: 51.5274, -0.0777
- Brixton: 51.4613, -0.1156
- Hammersmith: 51.4927, -0.2248

Respond ONLY with a JSON object. Keys are tool names, values are argument dicts.
If a tool needs no arguments or defaults are fine, use empty dict {{}}.
If the user asks about a specific area, provide coordinates.
If user asks to compare two locations with the same tool, use keys like "get_traffic_flow__1" and "get_traffic_flow__2".

Example for "traffic near Tower Bridge":
{{"get_traffic_flow": {{"latitude": 51.5055, "longitude": -0.0754, "location_name": "Tower Bridge"}}}}

Example for "weather and tube status":
{{"get_current_weather": {{}}, "get_tube_status": {{}}}}

Example for "compare traffic Canary Wharf vs City":
{{"get_traffic_flow__1": {{"latitude": 51.5054, "longitude": -0.0235, "location_name": "Canary Wharf"}}, "get_traffic_flow__2": {{"latitude": 51.5155, "longitude": -0.0922, "location_name": "City of London"}}}}

User question: {question}
JSON arguments:"""

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

DIRECT_RESPONSE_PROMPT = """You are the Smart City AI Agent for London. The user has sent a message
that doesn't require any data tools (it might be a greeting, a general question,
or something unrelated to city data).

Respond naturally. If they're greeting you, introduce yourself briefly and mention
what you can help with (London traffic, weather, air quality, tube status).
Keep it concise and friendly.

User message: {question}"""


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
    Returns NONE for simple greetings/non-city queries.
    """
    logger.info("🔀 Router: Deciding which tools to call...")

    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    try:
        llm = _get_llm()
        prompt = ROUTER_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        raw = raw.replace("`", "").replace('"', "").replace("'", "")

        # Check for NONE response (no tools needed)
        if raw.upper() == "NONE":
            logger.info("🔀 Router: No tools needed (greeting/non-city query)")
            return {
                "tools_to_call": [],
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        tool_names = [t.strip() for t in raw.split(",") if t.strip()]
        valid_tools = [t for t in tool_names if t in TOOL_MAP]

        if not valid_tools:
            logger.warning(f"Router returned no valid tools from: {raw}")
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


def argument_extractor_node(state: CityAgentState) -> dict:
    """
    NODE 2: Argument Extractor
    Determines what arguments to pass to each tool based on the user's question.
    E.g., if user asks about "Tower Bridge traffic", passes Tower Bridge coordinates.
    """
    tools_to_call = state.get("tools_to_call", [])
    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    # For simple single-tool queries with no location, skip LLM call
    no_arg_tools = {
        "get_tube_status", "get_road_disruptions",
        "get_london_traffic_overview", "get_traffic_incidents",
    }
    if all(t in no_arg_tools for t in tools_to_call):
        logger.info("🎯 Argument Extractor: All tools use defaults, skipping LLM call")
        tool_args = {t: {} for t in tools_to_call}
        return {"tool_arguments": tool_args}

    try:
        llm = _get_llm()
        prompt = ARGUMENT_PROMPT.format(
            question=question,
            tools=", ".join(tools_to_call),
        )
        response = llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        # Clean markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]  # Remove first line
            raw = raw.rsplit("```", 1)[0]  # Remove last fence
            raw = raw.strip()

        import json
        tool_args = json.loads(raw)
        logger.info(f"🎯 Argument Extractor: {tool_args}")

        return {"tool_arguments": tool_args}

    except Exception as e:
        logger.warning(f"Argument extraction failed: {e}. Using defaults.")
        tool_args = {t: {} for t in tools_to_call}
        return {"tool_arguments": tool_args}


def tool_executor_node(state: CityAgentState) -> dict:
    """
    NODE 3: Parallel Tool Executor
    Calls all selected tools SIMULTANEOUSLY using ThreadPoolExecutor.
    This is the key Day 5 upgrade — 3 API calls that took 3-4 seconds
    sequentially now complete in ~1.5 seconds.
    """
    tool_args = state.get("tool_arguments", {})
    tools_to_call = state.get("tools_to_call", [])

    # Build the execution plan: merge tool_args with tools_to_call
    execution_plan = {}
    if tool_args:
        # Use tool_arguments as the plan (may have __1, __2 suffixes)
        execution_plan = tool_args
    else:
        execution_plan = {t: {} for t in tools_to_call}

    total = len(execution_plan)
    logger.info(f"🔧 Tool Executor: Calling {total} tools in parallel...")

    results: dict[str, str] = {}

    def _call_tool(tool_key: str, args: dict) -> tuple[str, str]:
        """Call a single tool and return (key, result_string)."""
        # Handle suffixed keys like "get_traffic_flow__1"
        base_name = tool_key.split("__")[0]
        tool_func = TOOL_MAP.get(base_name)

        if not tool_func:
            return tool_key, f"ERROR: Unknown tool '{base_name}'"

        try:
            logger.info(f"  📡 Calling {tool_key} with args: {args}")
            result = tool_func.invoke(args if args else {})
            logger.info(f"  ✅ {tool_key} returned data")
            return tool_key, result
        except Exception as e:
            logger.error(f"  ❌ {tool_key} failed: {e}")
            return tool_key, f"ERROR: {str(e)}"

    # Execute all tools in parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(_call_tool, tool_key, args): tool_key
            for tool_key, args in execution_plan.items()
        }

        for future in as_completed(futures):
            tool_key = futures[future]
            try:
                key, result = future.result()
                results[key] = result
            except Exception as e:
                results[tool_key] = f"ERROR: {str(e)}"

    return {"tool_results": results}


def analyzer_node(state: CityAgentState) -> dict:
    """
    NODE 4: Analyzer
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
        fallback = "I collected the following data but couldn't generate a full analysis:\n\n"
        fallback += tool_data
        return {
            "analysis": fallback,
            "error": f"Analysis error: {str(e)}",
        }


def responder_node(state: CityAgentState) -> dict:
    """
    NODE 5: Responder
    Formats the analysis as the final AI message with data sources.
    """
    logger.info("💬 Responder: Formatting final answer...")

    analysis = state.get("analysis", "No analysis available.")
    tool_results = state.get("tool_results", {})

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


def direct_responder_node(state: CityAgentState) -> dict:
    """
    NODE (alternative): Direct Responder
    Handles greetings and non-city queries WITHOUT calling any tools.
    Saves 1 Gemini API call (no analyzer needed).
    """
    logger.info("💬 Direct Responder: Handling non-tool query...")

    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    try:
        llm = _get_llm()
        prompt = DIRECT_RESPONSE_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

    except Exception as e:
        logger.error(f"Direct responder error: {e}")
        text = (
            "Hello! I'm the Smart City AI Agent for London. "
            "I can help you with real-time traffic conditions, tube status, "
            "weather, and air quality. What would you like to know?"
        )

    return {
        "messages": [AIMessage(content=text)],
    }


# ══════════════════════════════════════════════════════════════════
# Conditional Edge: Should We Use Tools?
# ══════════════════════════════════════════════════════════════════

def should_use_tools(state: CityAgentState) -> str:
    """
    Conditional edge after router.
    If router selected tools → go to argument_extractor → tool_executor
    If router returned empty → go to direct_responder (skip tools entirely)
    """
    tools = state.get("tools_to_call", [])
    if tools:
        return "argument_extractor"
    return "direct_responder"


# ══════════════════════════════════════════════════════════════════
# Graph Builder
# ══════════════════════════════════════════════════════════════════

def build_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent with parallel execution
    and conditional routing.

    Flow:
      START → router → [should_use_tools?]
                          ├── yes → argument_extractor → tool_executor (parallel) → analyzer → responder → END
                          └── no  → direct_responder → END
    """
    graph = StateGraph(CityAgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("argument_extractor", argument_extractor_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("responder", responder_node)
    graph.add_node("direct_responder", direct_responder_node)

    # Entry point
    graph.add_edge(START, "router")

    # Conditional: does the query need tools?
    graph.add_conditional_edges(
        "router",
        should_use_tools,
        {
            "argument_extractor": "argument_extractor",
            "direct_responder": "direct_responder",
        },
    )

    # Tool path: extract args → execute → analyze → respond
    graph.add_edge("argument_extractor", "tool_executor")
    graph.add_edge("tool_executor", "analyzer")
    graph.add_edge("analyzer", "responder")
    graph.add_edge("responder", END)

    # Direct path: respond without tools
    graph.add_edge("direct_responder", END)

    # Compile
    agent = graph.compile()
    logger.info("✅ Agent graph compiled (parallel execution + conditional routing)")

    return agent


# ── Convenience function ──────────────────────────────────────────

def create_agent():
    """Create and return a ready-to-use agent instance."""
    return build_agent_graph()

"""
Smart City AI Agent - LangGraph Agent (Day 6)
Parallel execution + conditional routing + argument extraction + conversation memory.

Graph flow:
  START → router → [should_use_tools?]
                      ├── yes → argument_extractor → tool_executor (parallel) → analyzer → responder → END
                      └── no  → direct_responder → END
"""

import json
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
MAX_PARALLEL_WORKERS = 4

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

ROUTER_PROMPT = """Based on the user's question and conversation history, decide which tools to call.

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
- If the user is just greeting you or asking a non-city question, respond with: NONE
- Otherwise, return a comma-separated list of tool names. Choose the minimum set needed.
- For follow-up questions, consider the CONVERSATION HISTORY to understand context.
  Example: if user previously asked about traffic and now says "what about the weather?",
  they want weather for the same area.
- For broad questions ("how's London?"), use: get_tube_status,get_current_weather,get_london_traffic_overview

{conversation_context}

Current question: {question}

Tools to call (comma-separated list, or NONE):"""

ARGUMENT_PROMPT = """You are deciding what arguments to pass to city data tools.

User question: {question}
Tools to call: {tools}

{conversation_context}

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
If the user asks about a specific area (or references one from conversation history), provide coordinates.
If user asks to compare two locations with the same tool, use keys like "get_traffic_flow__1" and "get_traffic_flow__2".

User question: {question}
JSON arguments:"""

ANALYSIS_PROMPT = """You are analyzing real-time London city data to answer the user's question.

{conversation_context}

Current question: {question}

Data collected from tools:
{tool_data}

Provide a clear, insightful analysis that:
1. Directly answers the user's question
2. Cites specific numbers and data points
3. Identifies any correlations between data sources (e.g., weather affecting traffic)
4. Notes anything unusual or noteworthy
5. Is concise — lead with the key insight, then supporting details
6. If this is a follow-up question, reference relevant information from the conversation history

If some data sources failed or returned no data, work with what's available and note the gap briefly."""

DIRECT_RESPONSE_PROMPT = """You are the Smart City AI Agent for London. The user has sent a message
that doesn't require any data tools (it might be a greeting, a general question,
or something unrelated to city data).

{conversation_context}

Respond naturally. If they're greeting you, introduce yourself briefly and mention
what you can help with (London traffic, weather, air quality, tube status).
If there's conversation history, you can reference it.
Keep it concise and friendly.

User message: {question}"""


# ══════════════════════════════════════════════════════════════════
# LLM Instance
# ══════════════════════════════════════════════════════════════════

def _get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
        max_output_tokens=1024,
    )


# ══════════════════════════════════════════════════════════════════
# Conversation Context Helper
# ══════════════════════════════════════════════════════════════════

def _build_conversation_context(messages: list) -> str:
    """
    Build a conversation context string from message history.
    Only includes previous exchanges (not the current question).
    Returns empty string if no history.
    """
    # Filter to only LangChain message objects (skip tuples from initial invoke)
    history = []
    for msg in messages:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type == "human":
                history.append(f"User: {msg.content}")
            elif msg.type == "ai":
                # Truncate long AI responses for context
                content = msg.content
                if len(content) > 300:
                    content = content[:300] + "..."
                history.append(f"Agent: {content}")

    # Exclude the last user message (that's the current question)
    if history and history[-1].startswith("User:"):
        history = history[:-1]

    if not history:
        return ""

    context = "CONVERSATION HISTORY (previous exchanges):\n"
    context += "\n".join(history[-6:])  # Last 3 exchanges max
    return context


# ══════════════════════════════════════════════════════════════════
# Graph Nodes
# ══════════════════════════════════════════════════════════════════

def router_node(state: CityAgentState) -> dict:
    """
    NODE 1: Router — now context-aware.
    Sees conversation history to understand follow-up questions.
    """
    logger.info("🔀 Router: Deciding which tools to call...")

    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    conversation_context = _build_conversation_context(messages)

    try:
        llm = _get_llm()
        prompt = ROUTER_PROMPT.format(
            question=question,
            conversation_context=conversation_context,
        )
        response = llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        raw = raw.replace("`", "").replace('"', "").replace("'", "")

        if raw.upper() == "NONE":
            logger.info("🔀 Router: No tools needed")
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
    NODE 2: Argument Extractor — now uses conversation context
    to infer locations from follow-up questions.
    """
    tools_to_call = state.get("tools_to_call", [])
    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    no_arg_tools = {
        "get_tube_status", "get_road_disruptions",
        "get_london_traffic_overview", "get_traffic_incidents",
    }
    if all(t in no_arg_tools for t in tools_to_call):
        logger.info("🎯 Argument Extractor: All tools use defaults, skipping LLM call")
        tool_args = {t: {} for t in tools_to_call}
        return {"tool_arguments": tool_args}

    conversation_context = _build_conversation_context(messages)

    try:
        llm = _get_llm()
        prompt = ARGUMENT_PROMPT.format(
            question=question,
            tools=", ".join(tools_to_call),
            conversation_context=conversation_context,
        )
        response = llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

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
    Calls all selected tools simultaneously.
    """
    tool_args = state.get("tool_arguments", {})
    tools_to_call = state.get("tools_to_call", [])

    execution_plan = tool_args if tool_args else {t: {} for t in tools_to_call}

    total = len(execution_plan)
    logger.info(f"🔧 Tool Executor: Calling {total} tools in parallel...")

    results: dict[str, str] = {}

    def _call_tool(tool_key: str, args: dict) -> tuple[str, str]:
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
    NODE 4: Analyzer — now sees conversation history for
    context-aware analysis.
    """
    logger.info("🧠 Analyzer: Correlating data...")

    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    tool_results = state.get("tool_results", {})
    conversation_context = _build_conversation_context(messages)

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
            conversation_context=conversation_context,
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
    """NODE 5: Responder — formats the final answer."""
    logger.info("💬 Responder: Formatting final answer...")

    analysis = state.get("analysis", "No analysis available.")
    tool_results = state.get("tool_results", {})

    response_parts = [analysis]

    successful_tools = [t for t, r in tool_results.items() if not r.startswith("ERROR")]
    if successful_tools:
        sources = ", ".join(successful_tools)
        response_parts.append(f"\n\n📊 *Data sources: {sources}*")

    final_response = "\n".join(response_parts)
    return {"messages": [AIMessage(content=final_response)]}


def direct_responder_node(state: CityAgentState) -> dict:
    """NODE (alt): Direct response for non-tool queries."""
    logger.info("💬 Direct Responder: Handling non-tool query...")

    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    conversation_context = _build_conversation_context(messages)

    try:
        llm = _get_llm()
        prompt = DIRECT_RESPONSE_PROMPT.format(
            question=question,
            conversation_context=conversation_context,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

    except Exception as e:
        logger.error(f"Direct responder error: {e}")
        text = (
            "Hello! I'm the Smart City AI Agent for London. "
            "I can help you with real-time traffic conditions, tube status, "
            "weather, and air quality. What would you like to know?"
        )

    return {"messages": [AIMessage(content=text)]}


# ══════════════════════════════════════════════════════════════════
# Conditional Edge
# ══════════════════════════════════════════════════════════════════

def should_use_tools(state: CityAgentState) -> str:
    tools = state.get("tools_to_call", [])
    if tools:
        return "argument_extractor"
    return "direct_responder"


# ══════════════════════════════════════════════════════════════════
# Graph Builder
# ══════════════════════════════════════════════════════════════════

def build_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent.

    Flow:
      START → router → [should_use_tools?]
                          ├── yes → argument_extractor → tool_executor → analyzer → responder → END
                          └── no  → direct_responder → END
    """
    graph = StateGraph(CityAgentState)

    graph.add_node("router", router_node)
    graph.add_node("argument_extractor", argument_extractor_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("responder", responder_node)
    graph.add_node("direct_responder", direct_responder_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        should_use_tools,
        {
            "argument_extractor": "argument_extractor",
            "direct_responder": "direct_responder",
        },
    )
    graph.add_edge("argument_extractor", "tool_executor")
    graph.add_edge("tool_executor", "analyzer")
    graph.add_edge("analyzer", "responder")
    graph.add_edge("responder", END)
    graph.add_edge("direct_responder", END)

    agent = graph.compile()
    logger.info("✅ Agent graph compiled (parallel + conditional + memory-aware)")

    return agent


def create_agent():
    """Create and return a ready-to-use agent instance."""
    return build_agent_graph()

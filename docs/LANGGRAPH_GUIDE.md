# LangGraph Crash Course — What You Need for Day 4

## The Mental Model

Think of LangGraph as building a **flowchart that actually runs**. Each box in the flowchart is a "node" (a function), and each arrow is an "edge" (a transition). The flowchart passes a "state" object from node to node, and each node can read + modify it.

```
User Query
    │
    ▼
┌─────────────┐
│   ROUTER    │  ← Decides which tools to call
│   (node)    │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│     PARALLEL TOOL CALLS      │  ← Calls TfL + Weather + AQ simultaneously
│  ┌─────┐ ┌───────┐ ┌────┐   │
│  │ TfL │ │Weather│ │ AQ │   │
│  └──┬──┘ └───┬───┘ └─┬──┘   │
│     └────────┼───────┘       │
└──────────────┼───────────────┘
               │
               ▼
       ┌───────────────┐
       │   ANALYZER    │  ← LLM correlates all results
       │   (node)      │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │   RESPONSE    │  ← Formats final answer
       │   (node)      │
       └───────────────┘
```

## The 4 Core Concepts

### 1. State (TypedDict)

State is a dictionary that flows through the entire graph. Every node reads from it and writes to it.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The conversation so far
    messages: Annotated[list, add_messages]
    # Which tools the agent decided to call
    tools_to_call: list[str]
    # Results from each tool
    tool_results: dict[str, str]
    # Final analysis
    analysis: str
```

The `Annotated[list, add_messages]` is special — it means "when a node returns new messages, APPEND them to the list instead of replacing." This is how conversation history works.

### 2. Nodes (functions)

A node is just a Python function that takes state and returns updates to state:

```python
def router_node(state: AgentState) -> dict:
    """Decide which tools to call based on the user's question."""
    # LLM reads the latest message and picks tools
    user_message = state["messages"][-1].content
    # ... LLM call to decide tools ...
    return {"tools_to_call": ["tfl", "weather"]}

def analyze_node(state: AgentState) -> dict:
    """LLM analyzes all tool results and generates insight."""
    results = state["tool_results"]
    # ... LLM call to correlate data ...
    return {"analysis": "Traffic is bad because of rain + accident on A40"}
```

Key: nodes return a PARTIAL dict — only the keys they want to update. LangGraph merges it into the full state.

### 3. Edges (transitions)

Edges connect nodes. There are two types:

**Normal edge**: Always goes to the next node.
```python
graph.add_edge("router", "call_tools")
graph.add_edge("call_tools", "analyze")
```

**Conditional edge**: Picks the next node based on state.
```python
def should_continue(state: AgentState) -> str:
    """Decide if we need more data or can respond."""
    if state["tool_results"]:
        return "analyze"
    else:
        return "call_tools"

graph.add_conditional_edges("router", should_continue)
```

### 4. The Graph (putting it together)

```python
from langgraph.graph import StateGraph, START, END

# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("router", router_node)
graph.add_node("call_tools", call_tools_node)
graph.add_node("analyze", analyze_node)
graph.add_node("respond", respond_node)

# Add edges
graph.add_edge(START, "router")          # Entry point
graph.add_edge("router", "call_tools")   # Router → Tools
graph.add_edge("call_tools", "analyze")  # Tools → Analysis
graph.add_edge("analyze", "respond")     # Analysis → Response
graph.add_edge("respond", END)           # Exit

# Compile into a runnable
agent = graph.compile()

# Run it
result = agent.invoke({
    "messages": [("user", "Why is traffic bad in Central London?")],
    "tools_to_call": [],
    "tool_results": {},
    "analysis": "",
})
```

## How This Maps to OUR Agent

Here's exactly what we'll build on Day 4-5:

### Our State
```python
class CityAgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    query_type: str          # "traffic", "weather", "multi", "general"
    tools_needed: list[str]  # ["tfl", "weather", "air_quality", "tomtom"]
    tool_results: dict       # {"tfl": ToolResponse, "weather": ToolResponse, ...}
    analysis: str            # Correlated insight from LLM
    confidence: float        # How confident the agent is
    error: str | None        # Error if something went wrong
```

### Our Nodes
1. **Router**: LLM reads user question → decides which tools to call
2. **Tool Executor**: Calls selected API wrappers (parallel on Day 5!)
3. **Analyzer**: LLM sees all tool results → correlates patterns
4. **Response Formatter**: Structures the final answer with sources

### Our Edges
```
START → router → tool_executor → analyzer → response → END
                      ↑                         │
                      └─── needs_more_data ─────┘
                      (conditional: if analysis is incomplete)
```

## LangGraph vs AgentExecutor — The Key Difference

**AgentExecutor** (the old way):
- LLM is in a LOOP: think → act → observe → think → act → observe...
- You have NO control over the order of operations
- The LLM might call tools in a bad order or forget to call one

**LangGraph** (what we're using):
- YOU define the flow: router → tools → analyze → respond
- You control which tools run in parallel
- You can add checkpoints, retries, human-in-the-loop
- The LLM still makes DECISIONS, but within YOUR structure

Analogy: AgentExecutor is like giving someone a map and saying "find your way."
LangGraph is like building the roads and traffic lights, then letting them drive.

## What LangGraph Gives Us for Free

1. **Parallel execution**: Call TfL + Weather + AQ at the same time (Day 5)
2. **State persistence**: Built-in memory across conversation turns (Day 6)
3. **Streaming**: Stream node-by-node so the UI shows "Checking traffic..." → "Checking weather..." (Day 11)
4. **Conditional routing**: If the user asks about just weather, skip TfL tools entirely
5. **Error recovery**: If one tool fails, continue with the others

## Package Install (for Day 4)

```
pip install langgraph langchain-core langchain-google-genai
```

## Resources

- LangGraph docs: https://langchain-ai.github.io/langgraph/
- LangGraph tutorials: https://langchain-ai.github.io/langgraph/tutorials/
- Concept guide: https://langchain-ai.github.io/langgraph/concepts/
- "Functions, Tools and Agents with LangChain" on DeepLearning.ai

## Day 4 Preview

Tomorrow we will:
1. Install LangGraph + langchain-google-genai
2. Define CityAgentState
3. Convert all 4 tool wrappers into LangChain Tools
4. Build a basic linear graph: router → tools → analyze → respond
5. Test with a real query: "What's the traffic like in Central London?"

The key thing to understand tonight: **state flows through nodes, each node
does one job, edges connect them.** That's it. Everything else is details.

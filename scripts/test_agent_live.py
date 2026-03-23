"""
Smart City AI Agent - Day 5 Live Test Script
Tests parallel execution + conditional routing with real Gemini calls.

Usage: python -m scripts.test_agent_live

⚠️ Uses ~5-6 Gemini API calls total.
"""

import logging
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.agent.graph import create_agent


def test_query(agent, question: str, expect_tools: bool = True):
    """Run a single query and show timing."""
    print(f"\n{'='*70}")
    print(f"📨 Question: {question}")
    print(f"{'='*70}")

    start = time.perf_counter()
    result = agent.invoke({
        "messages": [("user", question)],
        "tools_to_call": [],
        "tool_arguments": {},
        "tool_results": {},
        "analysis": "",
        "iteration_count": 0,
        "error": "",
    })
    elapsed = time.perf_counter() - start

    # Tools selected
    tools = result.get("tools_to_call", [])
    print(f"\n🔧 Tools selected: {', '.join(tools) if tools else 'NONE (direct response)'}")

    # Tool arguments
    tool_args = result.get("tool_arguments", {})
    if tool_args:
        for tool_name, args in tool_args.items():
            if args:
                print(f"  🎯 {tool_name}: {args}")

    # Tool results (truncated)
    for tool_name, tool_result in result.get("tool_results", {}).items():
        truncated = tool_result[:120] + "..." if len(tool_result) > 120 else tool_result
        status = "✅" if not tool_result.startswith("ERROR") else "❌"
        print(f"  {status} {tool_name}: {truncated}")

    # Final response
    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    if ai_messages:
        print(f"\n💬 Response:\n{ai_messages[-1].content}")

    print(f"\n⏱️ Total time: {elapsed:.2f}s")

    if result.get("error"):
        print(f"⚠️ Error: {result['error']}")

    return elapsed


def main():
    print("🏙️ Smart City AI Agent — Day 5 Live Test")
    print("Features: Parallel execution + Conditional routing + Argument extraction\n")

    agent = create_agent()
    print("✅ Agent created\n")

    # Test 1: Direct response (no tools, ~1 Gemini call)
    print("\n" + "─"*70)
    print("TEST 1: Greeting (should skip tools entirely)")
    print("─"*70)
    t1 = test_query(agent, "Hello! What can you help me with?", expect_tools=False)

    # Test 2: Multi-tool query (parallel execution, ~3 Gemini calls)
    print("\n" + "─"*70)
    print("TEST 2: Multi-source query (should call tools in parallel)")
    print("─"*70)
    t2 = test_query(agent, "What's the traffic, weather, and tube status in London right now?")

    # Test 3: Location-specific query (tests argument extraction, ~3 Gemini calls)
    print("\n" + "─"*70)
    print("TEST 3: Location-specific query (should extract coordinates)")
    print("─"*70)
    t3 = test_query(agent, "How's the traffic near Canary Wharf?")

    print(f"\n{'='*70}")
    print(f"✅ All tests complete!")
    print(f"⏱️ Timing: Greeting={t1:.1f}s, Multi-tool={t2:.1f}s, Location={t3:.1f}s")
    print(f"💡 Multi-tool should be fast due to parallel API calls")
    print(f"📊 Total Gemini calls used: ~5-6 out of 250/day")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

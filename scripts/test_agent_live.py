"""
Smart City AI Agent - Quick Test Script
Run this manually to test the agent with a REAL Gemini API call.

Usage: python -m scripts.test_agent_live

⚠️ This uses 2 Gemini API calls (router + analyzer).
    You have 250 req/day on gemini-2.5-flash.
"""

import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.agent.graph import create_agent


def test_query(agent, question: str):
    """Run a single query through the agent."""
    print(f"\n{'='*70}")
    print(f"📨 Question: {question}")
    print(f"{'='*70}")

    result = agent.invoke({
        "messages": [("user", question)],
        "tools_to_call": [],
        "tool_results": {},
        "analysis": "",
        "iteration_count": 0,
        "error": "",
    })

    # Print tools selected
    tools = result.get("tools_to_call", [])
    print(f"\n🔧 Tools selected: {', '.join(tools)}")

    # Print tool results (truncated)
    for tool_name, tool_result in result.get("tool_results", {}).items():
        truncated = tool_result[:150] + "..." if len(tool_result) > 150 else tool_result
        print(f"  📡 {tool_name}: {truncated}")

    # Print final response
    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    if ai_messages:
        print(f"\n💬 Agent Response:\n{ai_messages[-1].content}")
    else:
        print(f"\n💬 Analysis:\n{result.get('analysis', 'No response')}")

    if result.get("error"):
        print(f"\n⚠️ Error: {result['error']}")


def main():
    print("🏙️ Smart City AI Agent — Live Test")
    print("This will use 2 Gemini API calls.\n")

    agent = create_agent()
    print("✅ Agent created successfully")

    # Test with ONE query to conserve API quota
    test_query(agent, "What's the current traffic and weather like in Central London?")

    print(f"\n{'='*70}")
    print("✅ Test complete! Used ~2 Gemini API calls.")
    print("💡 Run the server for more testing: uvicorn app.main:app --reload")
    print(f"   Then POST to: http://localhost:8000/api/agent/chat")


if __name__ == "__main__":
    main()

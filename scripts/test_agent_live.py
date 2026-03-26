"""
Smart City AI Agent - Day 6 Live Test: Conversation Memory
Tests multi-turn conversations where the agent remembers context.

Usage: python -m scripts.test_agent_live

⚠️ Uses ~8-10 Gemini API calls total.
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
from app.agent.sessions import SessionManager


def chat(session_manager, agent, session_id, question):
    """Simulate the /api/agent/chat flow."""
    session = session_manager.get_or_create_session(session_id)
    session.add_user_message(question)

    print(f"\n  👤 You: {question}")

    start = time.perf_counter()
    result = agent.invoke({
        "messages": session.get_recent_messages(max_messages=10),
        "tools_to_call": [],
        "tool_arguments": {},
        "tool_results": {},
        "analysis": "",
        "iteration_count": 0,
        "error": "",
    })
    elapsed = time.perf_counter() - start

    # Extract response
    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
    response_text = ai_messages[-1].content if ai_messages else result.get("analysis", "No response")

    tools = result.get("tools_to_call", [])
    session.add_ai_message(response_text)
    session.add_tools_used(tools)

    print(f"  🔧 Tools: {', '.join(tools) if tools else 'NONE'}")
    # Show first 300 chars of response
    display = response_text[:300] + "..." if len(response_text) > 300 else response_text
    print(f"  🤖 Agent: {display}")
    print(f"  ⏱️ {elapsed:.1f}s")

    return session.session_id


def main():
    print("🏙️ Smart City AI Agent — Day 6: Conversation Memory Test")
    print("=" * 70)

    agent = create_agent()
    sm = SessionManager()
    print("✅ Agent + Session Manager ready\n")

    # ── Test 1: Multi-turn conversation ───────────────────────────
    print("─" * 70)
    print("TEST 1: Multi-turn conversation (3 turns, same session)")
    print("The agent should remember previous context for follow-ups.")
    print("─" * 70)

    sid = None
    sid = chat(sm, agent, sid, "How's the traffic in Central London right now?")
    sid = chat(sm, agent, sid, "What about the weather there?")
    sid = chat(sm, agent, sid, "Is the air quality good?")

    # Show session stats
    session = sm.get_session(sid)
    print(f"\n  📝 Session {sid}: {session.metadata['total_queries']} queries, "
          f"tools used: {session.metadata['tools_used']}")

    # ── Test 2: New session (no context) ──────────────────────────
    print("\n" + "─" * 70)
    print("TEST 2: Fresh session (no history)")
    print("─" * 70)

    chat(sm, agent, None, "Hi! What can you do?")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"✅ Tests complete!")
    print(f"📊 Active sessions: {sm.active_count}")
    print(f"📊 Total Gemini calls used: ~8-10 out of 250/day")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

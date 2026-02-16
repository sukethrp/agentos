#!/usr/bin/env python3
"""
Simulation World Demo — stress-test an agent with 75 simulated customers.

This demo:
1. Creates a support-bot agent
2. Spins up a simulated world with 8 persona types
3. Fires 75 interactions using a "burst" traffic pattern
4. Evaluates every response (heuristic scoring — no extra API calls)
5. Prints a full report with pass rates, per-persona breakdown, and failures

Run:
    python examples/simulation_world_demo.py

To use the LLM judge (costs ~$0.02):
    OPENAI_API_KEY=sk-... python examples/simulation_world_demo.py --llm-judge
"""

from __future__ import annotations

import sys
from dotenv import load_dotenv

load_dotenv()

from agentos.core.agent import Agent
from agentos.simulation import (
    SimulatedWorld,
    WorldConfig,
    TrafficPattern,
    ALL_PERSONAS,
)


def make_demo_agent() -> Agent:
    """Create a mock-friendly support agent.

    For a real test you'd use a model like gpt-4o-mini, but for a
    quick local demo we use a minimal agent that always responds
    with a helpful template.  Replace this with a real Agent to
    stress-test against the OpenAI API.
    """

    class _MockMessage:
        def __init__(self, content: str):
            self.content = content
            self.tool_calls = None

    class MockAgent:
        """Deterministic mock agent for demo / CI use."""

        def __init__(self) -> None:
            self.config = type("C", (), {"name": "support-bot"})()
            self.events: list = []

        def run(self, user_input: str, stream: bool = False) -> _MockMessage:
            inp = (user_input or "").lower().strip()
            if not inp:
                return _MockMessage("I'm here to help! Could you please tell me what you need?")
            if any(w in inp for w in ["angry", "broken", "crash", "ridiculous", "worst", "refund", "cancel"]):
                return _MockMessage(
                    "I completely understand your frustration, and I sincerely apologize for the "
                    "inconvenience. Let me escalate this to our priority support team right away. "
                    "In the meantime, here are some steps that might help resolve the issue:\n"
                    "1. Clear your browser cache\n"
                    "2. Try logging out and back in\n"
                    "3. Check our status page at status.example.com\n"
                    "I'll personally follow up within the hour."
                )
            if any(w in inp for w in ["pricing", "price", "plan", "upgrade", "tier", "cost"]):
                return _MockMessage(
                    "Great question! We offer three plans:\n"
                    "- **Free**: Up to 1,000 requests/month\n"
                    "- **Pro** ($29/mo): 50,000 requests, priority support\n"
                    "- **Enterprise** (custom): Unlimited, SLA, dedicated manager\n"
                    "Would you like me to help you choose the right plan?"
                )
            if any(w in inp for w in ["confused", "lost", "don't understand", "don't know", "no idea", "help"]):
                return _MockMessage(
                    "No worries at all! Let me walk you through it step by step:\n"
                    "1. First, go to your Dashboard\n"
                    "2. Click 'Settings' in the sidebar\n"
                    "3. You'll see your options there\n"
                    "Would you like me to explain any of these in more detail?"
                )
            if any(w in inp for w in ["sla", "compliance", "soc2", "gdpr", "enterprise", "saml", "sso"]):
                return _MockMessage(
                    "For enterprise requirements:\n"
                    "- We offer 99.99% uptime SLA with our Enterprise plan\n"
                    "- SOC2 Type II certified\n"
                    "- GDPR compliant with EU data residency options\n"
                    "- SAML SSO integration available\n"
                    "I'd recommend scheduling a call with our enterprise team for detailed planning."
                )
            # Default helpful response
            return _MockMessage(
                "Thank you for reaching out! I'd be happy to help you with that. "
                "Here's what I'd suggest:\n"
                "1. Check our documentation at docs.example.com\n"
                "2. Try the getting-started guide\n"
                "3. Let me know if you need more specific help\n"
                "Is there anything else I can assist you with?"
            )

    return MockAgent()  # type: ignore[return-value]


def main() -> None:
    use_llm = "--llm-judge" in sys.argv

    print("🌐 AgentOS Simulation World Demo")
    print("=" * 60)
    print(f"  Personas: {len(ALL_PERSONAS)}")
    for p in ALL_PERSONAS:
        print(f"    • {p.name:<28} mood={p.mood.value:<10} difficulty={p.difficulty}")
    print()

    agent = make_demo_agent()

    config = WorldConfig(
        total_interactions=75,
        concurrency=8,
        traffic_pattern=TrafficPattern.BURST,
        requests_per_second=5.0,
        use_llm_judge=use_llm,
        pass_threshold=5.5,
        quiet=False,
    )

    world = SimulatedWorld(agent, config)
    report = world.run()

    # Print chart data summary
    print("\n📊 Score Distribution:")
    for bucket, count in report.score_distribution.items():
        bar = "█" * count
        print(f"  {bucket:>5}: {bar} ({count})")

    if report.worst_interactions:
        print("\n🔍 Worst Interactions (for debugging):")
        for w in report.worst_interactions[:5]:
            print(f"  [{w['id']:3d}] {w['persona']:<24} quality={w['overall']:.1f}")
            print(f"        query: {w['query'][:80]}...")
            if w.get("failure_reason"):
                print(f"        reason: {w['failure_reason']}")
            print()

    print(f"\n✅ Simulation complete. {report.total_interactions} interactions processed.")


if __name__ == "__main__":
    main()

"""Demo Mode — try the full AgentOS platform without any API keys.

This example shows how to use MockProvider for testing, demos, and
local development. No OPENAI_API_KEY or any other secrets required.

Usage::

    python examples/demo_mode.py

Or set the environment variable and use any AgentOS entry-point::

    AGENTOS_DEMO_MODE=true python examples/quickstart.py
    AGENTOS_DEMO_MODE=true python -m agentos web
"""

import os
import sys

# Activate demo mode before any AgentOS imports
os.environ["AGENTOS_DEMO_MODE"] = "true"

sys.path.insert(0, "src")

from agentos.core.agent import Agent
from agentos.core.tool import tool


# ── Define tools (same as quickstart) ────────────────────────────


@tool(description="Calculate a math expression like '2 + 2' or '100 * 3.14'")
def calculator(expression: str) -> str:
    try:
        from agentos.tools.safe_math import safe_eval_math
        return str(safe_eval_math(expression))
    except (ValueError, ZeroDivisionError, ArithmeticError) as e:
        return f"Error: {e}"


@tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    weather = {
        "boston": "45°F, Cloudy with light snow",
        "new york": "48°F, Partly cloudy",
        "san francisco": "62°F, Sunny",
        "tokyo": "55°F, Clear skies",
        "london": "50°F, Overcast with drizzle",
    }
    return weather.get(city.lower(), f"No weather data for {city}")


@tool(description="Look up company information")
def company_lookup(company_name: str) -> str:
    companies = {
        "anthropic": "AI safety company, makers of Claude, HQ: San Francisco, Founded: 2021",
        "openai": "AI research lab, makers of ChatGPT, HQ: San Francisco, Founded: 2015",
        "google": "Tech giant, makers of Gemini and Search, HQ: Mountain View, Founded: 1998",
    }
    return companies.get(company_name.lower(), f"No data for {company_name}")


# ── Build the agent ──────────────────────────────────────────────

agent = Agent(
    name="demo-agent",
    model="gpt-4o-mini",
    tools=[calculator, get_weather, company_lookup],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)


# ── Sample queries ───────────────────────────────────────────────

DEMO_QUERIES = [
    "What's the weather in San Francisco right now?",
    "Calculate a 15% tip on a $85.50 dinner bill",
    "Tell me about AgentOS and what it can do",
]


if __name__ == "__main__":
    print("=" * 60)
    print("🎭 AgentOS Demo Mode")
    print("   No API keys needed — using MockProvider")
    print("=" * 60)

    total_cost = 0.0
    total_tokens = 0

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"  Query {i}/{len(DEMO_QUERIES)}: {query}")
        print(f"{'─' * 60}")

        response = agent.run(query)

        run_cost = sum(e.cost_usd for e in agent.events)
        run_tokens = sum(e.tokens_used for e in agent.events)
        total_cost += run_cost
        total_tokens += run_tokens

    # ── Final summary ────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("📊 Demo Session Summary")
    print(f"{'=' * 60}")
    print(f"   Queries run:     {len(DEMO_QUERIES)}")
    print(f"   Total tokens:    {total_tokens:,}")
    print(f"   Total cost:      ${total_cost:.4f} (simulated)")
    print(f"   Provider:        MockProvider (no API keys)")
    print(f"{'=' * 60}")
    print()
    print("💡 To start the full web platform in demo mode:")
    print("   AGENTOS_DEMO_MODE=true python -m agentos web")

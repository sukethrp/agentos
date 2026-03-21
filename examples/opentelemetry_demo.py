"""
AgentOS OpenTelemetry Demo
===========================

Shows how to export traces and metrics from AgentOS to any
OpenTelemetry-compatible backend (Jaeger, Datadog, Grafana, etc.).

Prerequisites:
    pip install 'agentos-platform[otel]'

Run with console output (no collector needed):
    python examples/opentelemetry_demo.py

Run with an OTLP collector (e.g. Jaeger):
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
        python examples/opentelemetry_demo.py

Spans emitted:
    agent.run  (root)
    ├── llm.call        (per LLM invocation)
    ├── tool.execute    (per tool call)
    └── llm.call        (final synthesis)

Metrics emitted:
    agentos.agent.runs   — counter
    agentos.llm.tokens   — counter
    agentos.llm.cost     — counter (USD)
    agentos.llm.latency  — histogram (ms)
    agentos.tool.latency  — histogram (ms)
"""

import os
import sys

sys.path.insert(0, "src")

os.environ.setdefault("AGENTOS_DEMO_MODE", "true")

from agentos.telemetry import setup
from agentos.core.tool import tool
from agentos.core.agent import Agent


# ── 1. Configure OpenTelemetry ──────────────────────────────────────────

endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
setup(
    service_name="agentos-otel-demo",
    otlp_endpoint=endpoint,
    console_export=not endpoint,
)


# ── 2. Define tools ─────────────────────────────────────────────────────

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    from agentos.tools.safe_math import safe_eval_math
    return str(safe_eval_math(expression))


@tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    weather = {
        "boston": "45°F, Cloudy",
        "san francisco": "62°F, Sunny",
    }
    return weather.get(city.lower(), f"No data for {city}")


# ── 3. Create agent and run ─────────────────────────────────────────────

agent = Agent(
    name="otel-demo-agent",
    model="gpt-4o-mini",
    tools=[calculator, get_weather],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

if __name__ == "__main__":
    print("=" * 60)
    print("📡 AgentOS OpenTelemetry Demo")
    if endpoint:
        print(f"   Exporting to: {endpoint}")
    else:
        print("   Console export (set OTEL_EXPORTER_OTLP_ENDPOINT for OTLP)")
    print("=" * 60)
    print()

    agent.run("What's the weather in Boston?")
    print()
    agent.run("What's 15% tip on a $85.50 bill?")

    print()
    print("=" * 60)
    print("✅ Done — check your collector for traces and metrics")
    print("=" * 60)

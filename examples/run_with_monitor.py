"""Run agents with the monitoring dashboard."""

import sys
sys.path.insert(0, "src")

import threading
import time
import uvicorn
from agentos.core.tool import tool
from agentos.core.agent import Agent
from agentos.monitor.store import store
from agentos.monitor.server import app


# --- Tools ---

@tool(description="Calculate a math expression")
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
        "london": "40°F, Rainy",
    }
    return weather.get(city.lower(), f"No weather data for {city}")


@tool(description="Look up company information")
def company_lookup(company_name: str) -> str:
    companies = {
        "anthropic": "AI safety company, makers of Claude, HQ: San Francisco",
        "openai": "AI research lab, makers of ChatGPT, HQ: San Francisco",
        "vertex": "Vertex Pharmaceuticals, biotech/pharma, HQ: Boston",
        "google": "Tech giant, makers of Gemini AI, HQ: Mountain View",
    }
    return companies.get(company_name.lower(), f"No data for {company_name}")


# --- Create Agent ---

agent = Agent(
    name="research-bot",
    model="gpt-4o-mini",
    tools=[calculator, get_weather, company_lookup],
    system_prompt="You are a helpful research assistant. Use tools when needed.",
)


# --- Monkey-patch agent to send events to store ---

original_run = agent.run

def monitored_run(user_input: str):
    result = original_run(user_input)
    for event in agent.events:
        store.log_event(event)
    return result

agent.run = monitored_run


# --- Start server in background, then run queries ---

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS Monitor — Starting Dashboard")
    print("=" * 60)
    print()
    print("📊 Dashboard: http://localhost:8000")
    print("📡 API:       http://localhost:8000/api/overview")
    print()
    print("Open http://localhost:8000 in your browser NOW!")
    print()

    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    # Run some queries
    queries = [
        "What's the weather in Boston?",
        "Calculate 15% tip on $85.50",
        "Tell me about Anthropic",
        "What's the weather in Tokyo and San Francisco?",
        "What's 1000 * 1.07^5?",
        "Compare OpenAI and Google",
        "What's the weather in London?",
        "Calculate 20% of 250",
    ]

    for q in queries:
        agent.run(q)
        time.sleep(0.5)

    print()
    print("=" * 60)
    print("✅ All queries complete! Dashboard is still running.")
    print("📊 Go to http://localhost:8000 to see the results!")
    print("   Press Ctrl+C to stop.")
    print("=" * 60)

    # Keep server alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down.")
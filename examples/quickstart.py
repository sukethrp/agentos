import sys
sys.path.insert(0, "src")

from agentos.core.tool import tool
from agentos.core.agent import Agent


@tool(description="Calculate a math expression like '2 + 2' or '100 * 3.14'")
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math allowed"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    weather = {
        "boston": "45°F, Cloudy with light snow",
        "new york": "48°F, Partly cloudy",
        "san francisco": "62°F, Sunny",
        "tokyo": "55°F, Clear skies",
    }
    return weather.get(city.lower(), f"No weather data for {city}")


@tool(description="Look up company information")
def company_lookup(company_name: str) -> str:
    companies = {
        "anthropic": "AI safety company, makers of Claude, HQ: San Francisco",
        "openai": "AI research lab, makers of ChatGPT, HQ: San Francisco",
        "vertex": "Vertex Pharmaceuticals, biotech/pharma, HQ: Boston",
    }
    return companies.get(company_name.lower(), f"No data for {company_name}")


agent = Agent(
    name="research-assistant",
    model="gpt-4o-mini",
    tools=[calculator, get_weather, company_lookup],
    system_prompt="You are a helpful research assistant. Use tools to find accurate information.",
)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS v0.3.0 — Quickstart Demo")
    print("=" * 60)

    agent.run("What's the weather in Boston right now?")
    agent.run("What's 15% tip on a $85.50 dinner bill?")
    agent.run("Tell me about Anthropic")
"""AgentOS Real Tools Demo — Agents connected to REAL APIs."""

import sys
sys.path.insert(0, "src")

from agentos.core.agent import Agent
from agentos.core.tool import tool
from agentos.tools.http_tool import weather_tool, web_search_tool, news_tool, create_api_tool


# --- Create real tools ---
real_weather = weather_tool()
real_search = web_search_tool()
real_news = news_tool()

# --- Create a custom API tool ---
joke_api = create_api_tool(
    name="random_joke",
    description="Get a random programming joke",
    url="https://official-joke-api.appspot.com/random_joke",
    method="GET",
)

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math allowed"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# --- Create Agent with REAL tools ---

agent = Agent(
    name="real-world-agent",
    model="gpt-4o-mini",
    tools=[real_weather, real_search, real_news, joke_api, calculator],
    system_prompt="You are a helpful assistant with access to real-time data. Use your tools to get current information. Be concise.",
)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS — Real-World Tools Demo")
    print("   (Connected to LIVE APIs!)")
    print("=" * 60)

    # Test 1: Real weather
    print("\n" + "━" * 60)
    print("TEST 1: Real Weather Data (Open-Meteo API)")
    print("━" * 60)
    agent.run("What's the actual current weather in Boston and Tokyo?")

    # Test 2: Web search
    print("\n" + "━" * 60)
    print("TEST 2: Web Search (DuckDuckGo)")
    print("━" * 60)
    agent.run("Search the web for information about AgentOS AI platform")

    # Test 3: Custom API
    print("\n" + "━" * 60)
    print("TEST 3: Custom API Tool (Joke API)")
    print("━" * 60)
    agent.run("Tell me a random programming joke")

    # Test 4: Multi-tool with real data
    print("\n" + "━" * 60)
    print("TEST 4: Multi-Tool Real Query")
    print("━" * 60)
    agent.run("What's the weather in London? Also calculate what 20% tip on $150 dinner would be.")
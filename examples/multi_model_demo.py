"""AgentOS Multi-Model Demo — Same agent, different brains."""

import sys
sys.path.insert(0, "src")

from agentos.core.tool import tool
from agentos.core.agent import Agent
from agentos.providers.router import list_providers


@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math allowed"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool(description="Get weather for a city")
def get_weather(city: str) -> str:
    weather = {
        "boston": "45°F, Cloudy with light snow",
        "tokyo": "55°F, Clear skies",
    }
    return weather.get(city.lower(), f"No data for {city}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS — Multi-Model Demo")
    print("=" * 60)

    # Show supported providers
    providers = list_providers()
    print("\n📋 Supported Providers:")
    for provider, models in providers.items():
        print(f"   {provider}: {', '.join(models[:3])}")

    query = "What's 20% tip on $75 and what's the weather in Tokyo?"

    # ── Test with OpenAI ──
    print("\n" + "━" * 60)
    print("🧠 MODEL 1: OpenAI GPT-4o-mini")
    print("━" * 60)

    openai_agent = Agent(
        name="openai-agent",
        model="gpt-4o-mini",
        tools=[calculator, get_weather],
    )
    openai_agent.run(query)

    # ── Test with Anthropic Claude ──
    print("\n" + "━" * 60)
    print("🧠 MODEL 2: Anthropic Claude Sonnet")
    print("━" * 60)

    try:
        claude_agent = Agent(
            name="claude-agent",
            model="claude-sonnet",
            tools=[calculator, get_weather],
        )
        claude_agent.run(query)
    except Exception as e:
        print(f"   ⚠️ Claude error: {e}")
        print("   (Make sure ANTHROPIC_API_KEY is set in .env)")

    # ── Test with Ollama (local) ──
    print("\n" + "━" * 60)
    print("🧠 MODEL 3: Ollama llama3.1 (Local — FREE)")
    print("━" * 60)

    try:
        ollama_agent = Agent(
            name="ollama-agent",
            model="ollama:llama3.1",
            tools=[calculator, get_weather],
        )
        ollama_agent.run(query)
    except ConnectionError:
        print("   ⚠️ Ollama not running. Start it with: ollama serve")
    except Exception as e:
        print(f"   ⚠️ Ollama error: {e}")

    print("\n" + "=" * 60)
    print("✅ Multi-model demo complete!")
    print("   Same tools, same query, different AI brains.")
    print("   Switch models with ONE line: model='gpt-4o-mini' → model='claude-sonnet'")
    print("=" * 60)
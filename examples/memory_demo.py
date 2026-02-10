"""AgentOS Memory Demo — Agents that remember."""

import sys
sys.path.insert(0, "src")

from agentos.core.tool import tool
from agentos.core.agent import Agent
from agentos.core.memory import Memory


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
        "san francisco": "62°F, Sunny",
    }
    return weather.get(city.lower(), f"No data for {city}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS — Memory Demo")
    print("=" * 60)

    # Create shared memory
    memory = Memory(enable_knowledge=True)

    # Pre-store some knowledge
    memory.store_fact("user_name", "Suketh", category="personal")
    memory.store_fact("user_role", "Founder of AgentOS", category="work")
    memory.store_fact("user_location", "Boston", category="personal")

    agent = Agent(
        name="memory-bot",
        model="gpt-4o-mini",
        tools=[calculator, get_weather],
        system_prompt="You are a helpful personal assistant. Use what you know about the user to personalize responses. Be warm and friendly.",
        memory=memory,
    )

    # Conversation 1: Agent should know the user's name
    print("\n" + "━" * 60)
    print("TEST 1: Agent should know the user from memory")
    print("━" * 60)
    agent.run("Hi! What do you know about me?")

    # Conversation 2: Agent should use location knowledge
    print("\n" + "━" * 60)
    print("TEST 2: Agent should use stored location")
    print("━" * 60)
    agent.run("What's the weather where I live?")

    # Conversation 3: User shares new info → agent extracts and stores
    print("\n" + "━" * 60)
    print("TEST 3: Agent learns new facts from conversation")
    print("━" * 60)
    agent.run("I work at Vertex Pharmaceuticals and I love building AI products")

    # Conversation 4: Agent should remember what was just said
    print("\n" + "━" * 60)
    print("TEST 4: Agent remembers previous conversation")
    print("━" * 60)
    agent.run("What company did I just tell you about?")

    # Conversation 5: Math with context
    print("\n" + "━" * 60)
    print("TEST 5: Tool use with memory context")
    print("━" * 60)
    agent.run("If my salary is $120,000 and I save 20%, how much do I save per month?")

    # Show memory state
    memory.print_memory()
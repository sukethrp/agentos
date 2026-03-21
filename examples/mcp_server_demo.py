"""
AgentOS MCP Server Demo
========================

Expose AgentOS @tool functions as an MCP-compatible server so any MCP client
(Claude Desktop, Cursor, etc.) can discover and invoke them.

Run with:
    python examples/mcp_server_demo.py

Or point an MCP client at this script via its stdio transport config:
    {
      "mcpServers": {
        "agentos-demo": {
          "command": "python",
          "args": ["examples/mcp_server_demo.py"]
        }
      }
    }
"""

import sys

sys.path.insert(0, "src")

from agentos.core.tool import tool
from agentos.core.agent import Agent
from agentos.mcp import MCPServer


# ── Define tools using the standard AgentOS @tool decorator ──────────────


@tool(description="Calculate a math expression like '2 + 2' or '100 * 3.14'")
def calculator(expression: str) -> str:
    try:
        from agentos.tools.safe_math import safe_eval_math
        return str(safe_eval_math(expression))
    except (ValueError, ZeroDivisionError, ArithmeticError) as e:
        return f"Error: {e}"


@tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    weather_data = {
        "boston": "45°F, Cloudy with light snow",
        "new york": "48°F, Partly cloudy",
        "san francisco": "62°F, Sunny",
        "tokyo": "55°F, Clear skies",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


@tool(description="Look up company information")
def company_lookup(company_name: str) -> str:
    companies = {
        "anthropic": "AI safety company, makers of Claude. HQ: San Francisco",
        "openai": "AI research lab, makers of ChatGPT. HQ: San Francisco",
    }
    return companies.get(company_name.lower(), f"No data for {company_name}")


# ── Option A: Build MCPServer directly from tools ───────────────────────

def run_from_tools():
    server = MCPServer("agentos-demo", tools=[calculator, get_weather, company_lookup])
    server.run()


# ── Option B: Build MCPServer from an existing Agent ────────────────────

def run_from_agent():
    agent = Agent(
        name="research-assistant",
        model="gpt-4o-mini",
        tools=[calculator, get_weather, company_lookup],
        system_prompt="You are a helpful research assistant.",
    )
    server = agent.as_mcp_server()
    server.run()


if __name__ == "__main__":
    run_from_tools()

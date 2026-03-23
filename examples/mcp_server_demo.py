"""Example: Run AgentOS tools as an MCP server.

This allows Claude Desktop, Cursor, or any MCP client to use
AgentOS tools like calculator, weather, RAG search, etc.

To use:
1. Run this script: python examples/mcp_server_demo.py
2. Or add to claude_desktop_config.json:
     {
         "mcpServers": {
             "agentos": {
                 "command": "python",
                 "args": ["examples/mcp_server_demo.py"]
             }
         }
     }
"""

from agentos.core.tool import tool
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


def run_from_tools():
    server = MCPServer(
        name="agentos-demo",
        version="0.3.0",
        tools=[calculator, get_weather, company_lookup],
    )
    server.run()


if __name__ == "__main__":
    run_from_tools()

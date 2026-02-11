"""Research Assistant Agent Template."""

from agentos.core.agent import Agent
from agentos.core.tool import tool
from agentos.tools.http_tool import web_search_tool, weather_tool, news_tool


def create_research_agent(model: str = "gpt-4o-mini", **kwargs) -> Agent:
    @tool(description="Calculate math expressions for data analysis")
    def calculator(expression: str) -> str:
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: Only basic math allowed"
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    return Agent(
        name="research-agent",
        model=model,
        tools=[web_search_tool(), news_tool(), weather_tool(), calculator],
        system_prompt="""You are a thorough research assistant. Follow these principles:
1. Always search for current information before answering
2. Cross-reference multiple sources when possible
3. Clearly distinguish between facts and opinions
4. Provide sources for your claims
5. If data is unavailable, say so honestly
6. Present findings in a clear, structured format
7. Use calculations to support quantitative claims
8. Summarize key findings at the end""",
        **kwargs,
    )
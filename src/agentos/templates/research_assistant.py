"""Research Assistant Agent Template."""

from agentos.core.agent import Agent
from agentos.tools.http_tool import (
    web_search_tool,
    weather_tool,
    news_tool,
    calculator_tool,
)


def create_research_agent(model: str = "gpt-4o-mini", **kwargs) -> Agent:
    return Agent(
        name="research-agent",
        model=model,
        tools=[web_search_tool(), news_tool(), weather_tool(), calculator_tool()],
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

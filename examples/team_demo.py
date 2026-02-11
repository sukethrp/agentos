"""AgentOS Multi-Agent Team Demo — Agents working together."""

import sys
sys.path.insert(0, "src")

from agentos.core.agent import Agent
from agentos.core.tool import tool
from agentos.core.team import AgentTeam


# --- Tools ---

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math allowed"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool(description="Look up market data for a company or industry")
def market_data(query: str) -> str:
    data = {
        "ai agents market": "AI agents market: $7.63B in 2025, projected $183B by 2033, 49.6% CAGR",
        "langchain": "LangChain: Open-source, 80K+ GitHub stars, Series A funded, ~$20M ARR",
        "crewai": "CrewAI: Multi-agent framework, 20K+ stars, growing rapidly in enterprise",
        "openai revenue": "OpenAI: $5B+ ARR in 2025, 300M+ weekly active users",
        "ai market": "Global AI market: $244B in 2025, projected $827B by 2030",
    }
    return data.get(query.lower(), f"Market data not available for: {query}")


@tool(description="Search for competitor information")
def competitor_search(company: str) -> str:
    competitors = {
        "langchain": "LangChain: Python framework for LLM apps. Strengths: large ecosystem, many integrations. Weaknesses: complex, heavy, no built-in testing or governance.",
        "crewai": "CrewAI: Multi-agent orchestration. Strengths: easy team setup, role-based agents. Weaknesses: no monitoring, no governance, limited testing.",
        "autogen": "AutoGen (Microsoft): Multi-agent conversations. Strengths: backed by Microsoft, good for research. Weaknesses: complex setup, not production-ready.",
    }
    return competitors.get(company.lower(), f"No competitor data for: {company}")


# --- Create Specialist Agents ---

researcher = Agent(
    name="researcher",
    model="gpt-4o-mini",
    tools=[market_data, competitor_search],
    system_prompt="You are a market researcher. Find data, analyze trends, and provide factual insights. Be thorough and cite sources.",
)

analyst = Agent(
    name="analyst",
    model="gpt-4o-mini",
    tools=[calculator],
    system_prompt="You are a business analyst. Take research data and extract actionable insights. Calculate growth rates, market shares, and projections. Be quantitative.",
)

writer = Agent(
    name="writer",
    model="gpt-4o-mini",
    tools=[],
    system_prompt="You are a professional report writer. Take analysis and research, and write a clear, compelling executive summary. Use bullet points for key findings. Keep it under 300 words.",
)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS — Multi-Agent Team Demo")
    print("=" * 60)

    # ── Demo 1: Sequential Team ──
    print("\n" + "━" * 60)
    print("DEMO 1: Sequential Pipeline (Researcher → Analyst → Writer)")
    print("━" * 60)

    sequential_team = AgentTeam(
        name="research-pipeline",
        agents=[researcher, analyst, writer],
        strategy="sequential",
    )

    result = sequential_team.run(
        "Analyze the AI agent market opportunity for AgentOS. Who are the competitors, what's the market size, and what's our competitive advantage?"
    )

    # ── Demo 2: Debate Team ──
    print("\n" + "━" * 60)
    print("DEMO 2: Debate (Agents argue and refine their answers)")
    print("━" * 60)

    debater1 = Agent(
        name="optimist",
        model="gpt-4o-mini",
        tools=[market_data],
        system_prompt="You are a bullish analyst. Focus on growth opportunities and positive trends. Be persuasive but factual.",
    )

    debater2 = Agent(
        name="skeptic",
        model="gpt-4o-mini",
        tools=[market_data],
        system_prompt="You are a critical analyst. Challenge assumptions, identify risks, and stress-test ideas. Be rigorous.",
    )

    debate_team = AgentTeam(
        name="investment-debate",
        agents=[debater1, debater2],
        strategy="debate",
        max_rounds=2,
    )

    result = debate_team.run(
        "Should a startup invest in building an AI agent platform in 2026? Make the case for and against."
    )
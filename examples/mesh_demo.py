"""Agent Mesh Demo — multi-agent collaboration with AgentOS.

Demonstrates:
  1. Orchestrator pattern — a coordinator delegates to specialists
  2. Peer-to-peer pattern — agents call each other directly
  3. Shared context — agents read upstream results
  4. Aggregated cost tracking across the chain

Usage:
    # Set your API key (or use demo mode)
    export OPENAI_API_KEY=sk-...
    # Or: export AGENTOS_DEMO_MODE=true

    python examples/mesh_demo.py
"""

from agentos.core.agent import Agent
from agentos.core.tool import tool
from agentos.mesh import AgentMesh
from agentos.tools.safe_math import safe_eval_math


# ── Define specialist tools ──

@tool(description="Calculate a math expression safely")
def calculator(expression: str) -> str:
    return str(safe_eval_math(expression))


@tool(description="Look up a company's latest stock price (mock)")
def stock_price(ticker: str) -> str:
    prices = {
        "AAPL": 227.50, "GOOGL": 181.30, "MSFT": 430.15,
        "AMZN": 205.70, "NVDA": 142.60, "TSLA": 265.80,
    }
    price = prices.get(ticker.upper())
    if price:
        return f"{ticker.upper()}: ${price:.2f}"
    return f"Ticker {ticker} not found. Available: {', '.join(prices)}"


@tool(description="Get market news headlines (mock)")
def market_news(topic: str) -> str:
    return (
        f"Headlines for '{topic}':\n"
        "1. Tech stocks rally on strong earnings\n"
        "2. Fed signals steady rates through Q2 2026\n"
        "3. AI sector investment hits record $180B in 2025"
    )


# ── Create specialist agents ──

researcher = Agent(
    name="researcher",
    model="gpt-4o-mini",
    tools=[stock_price, market_news],
    system_prompt=(
        "You are a financial researcher. Look up stock prices and news. "
        "Return concise, factual summaries."
    ),
)

analyst = Agent(
    name="analyst",
    model="gpt-4o-mini",
    tools=[calculator],
    system_prompt=(
        "You are a financial analyst. When given data, perform calculations "
        "and provide insights. Be quantitative and precise."
    ),
)

writer = Agent(
    name="writer",
    model="gpt-4o-mini",
    tools=[],
    system_prompt=(
        "You are a report writer. Take research and analysis and produce "
        "a clear, well-structured report. Use markdown formatting."
    ),
)


def demo_orchestrated():
    """Demo 1: Orchestrator pattern — coordinator delegates to specialists."""
    print("\n" + "=" * 70)
    print("DEMO 1: Orchestrator Pattern")
    print("=" * 70)

    coordinator = Agent(
        name="coordinator",
        model="gpt-4o-mini",
        tools=[],
        system_prompt=(
            "You are a team coordinator. Break tasks into subtasks and "
            "delegate to the right specialist:\n"
            "  - 'researcher': looks up stock prices and market news\n"
            "  - 'analyst': performs financial calculations\n"
            "  - 'writer': writes polished reports\n"
            "Use the delegate tool to assign work. Combine results into "
            "a final answer."
        ),
    )

    mesh = AgentMesh(name="finance-team")
    mesh.add(researcher)
    mesh.add(analyst)
    mesh.add(writer)

    result = mesh.run_orchestrated(
        coordinator=coordinator,
        task=(
            "Get the stock prices for AAPL and GOOGL, calculate the combined "
            "market cap if AAPL has 15.2B shares and GOOGL has 5.9B shares, "
            "then write a brief 3-paragraph market summary."
        ),
    )

    print(f"\n📋 Final report:\n{result.content}")


def demo_peer_to_peer():
    """Demo 2: Peer-to-peer — any agent can delegate to any other."""
    print("\n" + "=" * 70)
    print("DEMO 2: Peer-to-Peer Pattern")
    print("=" * 70)

    mesh = AgentMesh(name="p2p-team")
    mesh.add(researcher)
    mesh.add(analyst)
    mesh.add(writer)

    mesh.enable_peer_delegation()

    result = researcher.run(
        "Look up the stock price for NVDA, then delegate to analyst "
        "to calculate a 10% price increase, then delegate to writer "
        "to create a one-paragraph investment note."
    )

    mesh.cost_tracker.print_summary()
    print(f"\n📋 Final result:\n{result.content}")

    mesh.disable_peer_delegation()


def demo_shared_context():
    """Demo 3: Shared context — agents see each other's outputs."""
    print("\n" + "=" * 70)
    print("DEMO 3: Shared Context")
    print("=" * 70)

    mesh = AgentMesh(name="context-team")
    mesh.add(researcher)
    mesh.add(analyst)
    mesh.add(writer)

    mesh.shared_context.set("report_style", "Executive briefing, max 200 words", author="user")
    mesh.shared_context.set("audience", "Board of Directors", author="user")

    research = mesh.delegate("user", "researcher", "Get MSFT stock price and latest market news on AI")
    mesh.shared_context.set("research_findings", research[:300], author="researcher")

    analysis = mesh.delegate("user", "analyst", "Based on the shared context, estimate MSFT revenue impact if AI grows 20%")
    mesh.shared_context.set("analysis", analysis[:300], author="analyst")

    report = mesh.delegate("user", "writer", "Write the executive briefing using all shared context")

    mesh.cost_tracker.print_summary()
    print(f"\n📋 Executive briefing:\n{report}")


if __name__ == "__main__":
    print("🔗 AgentOS Mesh Demo — Multi-Agent Collaboration")
    print("=" * 70)

    demo_orchestrated()
    demo_peer_to_peer()
    demo_shared_context()

"""AgentOS A/B Test Demo — compare two agent variants."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentos.core.agent import Agent  # type: ignore
from agentos.core.ab_testing import ABTest, clone_agent  # type: ignore
from agentos.tools import get_builtin_tools  # type: ignore


def divider(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    divider("🧪 AgentOS — A/B Testing Demo")

    tools = list(get_builtin_tools().values())

    # ── Create a base agent ──
    base_agent = Agent(
        name="base-agent",
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="You are a helpful assistant. Answer clearly in 2-3 sentences.",
        temperature=0.7,
    )

    # ── Clone and modify ──
    agent_v1 = clone_agent(base_agent, "agent-v1")
    agent_v2 = clone_agent(base_agent, "agent-v2")

    # Make v2 slightly more creative
    agent_v2.config.system_prompt = (
        "You are a creative assistant. Provide more detailed, example-rich answers "
        "in 3-5 sentences while staying accurate."
    )
    agent_v2.config.temperature = 1.0

    print("Created two variants:")
    print(f"  • {agent_v1.config.name}: temp={agent_v1.config.temperature}")
    print(f"  • {agent_v2.config.name}: temp={agent_v2.config.temperature}")

    # ── Define test queries ──
    queries = [
        "Summarize the benefits of using AgentOS.",
        "Explain the difference between GPT-4o and GPT-4o-mini.",
        "Give three ideas for onboarding flows for a SaaS dashboard.",
        "Help me debug why a Python script might be slow.",
        "Write a short product description for an AI agent platform.",
    ]

    divider("Step 2: Run A/B test with 5 queries")

    tester = ABTest(agent_v1, agent_v2)
    report = tester.run_test(queries, num_runs=2)

    report.print_report()

    divider("Summary")
    print(f"Winner: {report.winner}  (confidence {report.confidence*100:.1f}%)")
    a = report.scores["agent_a"]
    b = report.scores["agent_b"]
    print(
        f"Agent A ({report.agent_a_name}) — avg_overall={a.avg_overall:.2f}, "
        f"win_rate={a.win_rate*100:.1f}%"
    )
    print(
        f"Agent B ({report.agent_b_name}) — avg_overall={b.avg_overall:.2f}, "
        f"win_rate={b.win_rate*100:.1f}%"
    )

    print("\n✅ A/B test demo complete")


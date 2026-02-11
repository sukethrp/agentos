"""AgentOS Workflow Demo — Lead Processing pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentos.core.agent import Agent  # type: ignore
from agentos.tools import get_builtin_tools  # type: ignore
from agentos.workflows import Workflow, WorkflowRunner  # type: ignore


def divider(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    divider("🔀 AgentOS — Lead Processing Workflow Demo")

    tools = get_builtin_tools()
    web_search = tools["web_search"]

    # ── Define agents for each stage ──
    research_agent = Agent(
        name="research-agent",
        model="gpt-4o-mini",
        tools=[web_search],
        system_prompt="You are a B2B research assistant. Be concise and factual.",
    )

    qualify_agent = Agent(
        name="qualify-agent",
        model="gpt-4o-mini",
        tools=[],
        system_prompt=(
            "You are a sales qualification bot. Given company context, "
            "decide if the lead has budget over $10K and explain why. "
            "End with a line: 'Qualified: yes' or 'Qualified: no'."
        ),
    )

    schedule_agent = Agent(
        name="schedule-agent",
        model="gpt-4o-mini",
        tools=[],
        system_prompt="You are an SDR. Draft a short email to schedule a demo.",
    )

    nurture_agent = Agent(
        name="nurture-agent",
        model="gpt-4o-mini",
        tools=[],
        system_prompt="You are a nurture specialist. Draft a polite nurture email for smaller-budget leads.",
    )

    # ── Build workflow ──
    wf = (
        Workflow("lead-processing")
        .step(
            "research",
            agent=research_agent,
            query="Research the company {company_name} and summarize key facts in 5 bullet points.",
        )
        .step(
            "qualify",
            agent=qualify_agent,
            query="Given this research: {research}, decide if the lead has budget over $10K and explain.",
        )
        .condition(
            "is_qualified",
            lambda output: "qualified: yes" in output.lower(),
        )
        .step(
            "schedule_demo",
            agent=schedule_agent,
            query=(
                "Write a short email to schedule a 30-minute demo with {company_name}, "
                "referencing this context: {research}"
            ),
            when="is_qualified",
        )
        .step(
            "nurture_email",
            agent=nurture_agent,
            query=(
                "Write a friendly nurture email to {company_name}, explaining we may not be the best fit "
                "right now but offering helpful resources. Use this context: {research}"
            ),
            when_not="is_qualified",
        )
        .build()
    )

    # ── Run workflow ──
    runner = WorkflowRunner(wf)

    context = {
        "company_name": "Acme Analytics",
    }

    execution = runner.run(context)

    divider("Execution Summary")
    print(f"Workflow ID:   {execution.id}")
    print(f"Workflow Name: {execution.workflow_name}")
    print(f"Status:        {execution.status}")
    print(f"Path:          {' → '.join(execution.path)}")

    for name, result in execution.steps.items():
        print(f"\nStep: {name}  [{result.status}]")
        if result.error:
            print(f"  Error: {result.error}")
        else:
            print(f"  Output (first 200 chars): {result.output[:200]}")
            print(f"  Cost: ${result.cost:.4f} | Duration: {result.duration_ms:.0f}ms")

    print("\n✅ Lead processing workflow demo complete")


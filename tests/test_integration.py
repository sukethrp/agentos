"""
Integration test: Full AgentOS pipeline using MockProvider.

This test proves that the entire flow works end-to-end:
1. Create a GovernedAgent with MockProvider
2. Register tools
3. Run a query
4. Verify governance (budget tracking, permissions)
5. Run sandbox tests
6. Verify monitoring events were captured

This runs in CI without any API keys.
"""

import os
from unittest.mock import patch

import pytest

from agentos.core.agent import Agent
from agentos.core.tool import Tool, tool
from agentos.core.types import Message, Role
from agentos.governance.budget import BudgetGuard
from agentos.governance.permissions import PermissionGuard
from agentos.governed_agent import GovernedAgent
from agentos.monitor.store import AgentStore
from agentos.sandbox.scenario import Scenario, SandboxReport


def _calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


def _greeter(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _enable_demo_mode():
    """Enable demo mode for every test so the MockProvider is used."""
    with patch.dict(os.environ, {"AGENTOS_DEMO_MODE": "true"}):
        yield


class TestFullPipeline:
    """End-to-end tests using MockProvider."""

    def test_agent_runs_query_with_mock_provider(self):
        """Agent should process a query and return a response."""
        calc_tool = Tool(fn=_calculator, name="calculator", description="Math calculator")
        agent = GovernedAgent(
            name="test-agent",
            model="gpt-4o-mini",
            tools=[calc_tool],
            budget=BudgetGuard(max_per_day=10.00),
        )

        result = agent.run("What is 2 + 2?")

        assert isinstance(result, Message)
        assert result.content is not None
        assert len(result.content) > 0
        assert result.role == Role.ASSISTANT
        assert agent.total_cost > 0
        assert len(agent.agent.events) > 0

    def test_governance_blocks_over_budget(self):
        """Budget guard should block queries that exceed the limit."""
        calc_tool = Tool(fn=_calculator, name="calculator", description="Math calculator")
        tiny_budget = BudgetGuard(
            max_per_action=0.01,
            max_per_hour=0.02,
            max_per_day=0.02,
            max_total=0.02,
        )
        agent = GovernedAgent(
            name="budget-test-agent",
            model="gpt-4o-mini",
            tools=[calc_tool],
            budget=tiny_budget,
        )

        agent.run("Calculate 2 + 2")

        # Agent tracks LLM costs even when the budget guard only records
        # tool-level spend via record_spend.  Verify LLM cost was captured.
        assert agent.total_cost > 0

        # Simulate accumulated spend to push past the total budget ceiling,
        # then verify the guard blocks the next action.
        tiny_budget.record_spend(0.019)
        check = agent.governance.check_tool_call("calculator", estimated_cost=0.005)
        assert not check.allowed
        assert "budget" in check.rule or "exceed" in check.message.lower()

    def test_governance_blocks_forbidden_tool(self):
        """Permission guard should block tools in the blocked list."""
        def dangerous_tool(cmd: str) -> str:
            return "should never run"

        safe_tool = Tool(fn=_greeter, name="greeter", description="Greets people")
        blocked_tool = Tool(fn=dangerous_tool, name="dangerous_tool", description="Dangerous")

        agent = GovernedAgent(
            name="perm-test-agent",
            model="gpt-4o-mini",
            tools=[safe_tool, blocked_tool],
            permissions=PermissionGuard(blocked_tools=["dangerous_tool"]),
            budget=BudgetGuard(max_per_day=10.00),
        )

        result = agent.governance.check_tool_call("dangerous_tool", estimated_cost=0.001)
        assert not result.allowed
        assert "blocked" in result.message.lower() or "permission" in result.rule

        result_safe = agent.governance.check_tool_call("greeter", estimated_cost=0.001)
        assert result_safe.allowed

    def test_sandbox_scores_responses(self):
        """Sandbox should produce a report with scores between 0 and 10."""
        calc_tool = Tool(fn=_calculator, name="calculator", description="Math calculator")
        agent = Agent(
            name="sandbox-test-agent",
            model="gpt-4o-mini",
            tools=[calc_tool],
        )

        scenarios = [
            Scenario(
                name="Basic greeting",
                user_message="Hello!",
                expected_behavior="Responds politely",
                max_cost=1.00,
            ),
        ]

        def mock_judge(scenario, agent_response, tools_used):
            return {
                "relevance": 8.0,
                "quality": 7.0,
                "safety": 9.0,
                "reasoning": "Mock judge: response was appropriate",
            }

        with patch("agentos.sandbox.runner.judge_response", side_effect=mock_judge):
            from agentos.sandbox.runner import Sandbox

            sandbox = Sandbox(agent, pass_threshold=6.0, embedder=_StubEmbedder())
            report = sandbox.run(scenarios)

        assert isinstance(report, SandboxReport)
        assert report.total_scenarios == 1
        assert len(report.results) == 1

        result = report.results[0]
        assert 0 <= result.relevance_score <= 10
        assert 0 <= result.quality_score <= 10
        assert 0 <= result.safety_score <= 10
        assert 0 <= result.overall_score <= 10
        assert result.overall_score == round((8.0 + 7.0 + 9.0) / 3, 1)

    def test_tool_decorator_registers_correctly(self):
        """@tool decorator should capture name, description, params."""

        @tool(description="Add two numbers together")
        def add_numbers(a: int, b: int) -> str:
            return str(a + b)

        assert isinstance(add_numbers, Tool)
        assert add_numbers.name == "add_numbers"
        assert add_numbers.description == "Add two numbers together"

        param_names = [p.name for p in add_numbers.params]
        assert "a" in param_names
        assert "b" in param_names

        for p in add_numbers.params:
            if p.name in ("a", "b"):
                assert p.type == "number"
                assert p.required is True

        spec = add_numbers.spec
        assert spec.name == "add_numbers"
        assert len(spec.parameters) == 2

    def test_demo_mode_works_without_api_keys(self):
        """Demo mode should work with no API keys set."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            os.environ["AGENTOS_DEMO_MODE"] = "true"

            calc_tool = Tool(fn=_calculator, name="calculator", description="Math calculator")
            agent = GovernedAgent(
                name="demo-agent",
                model="gpt-4o-mini",
                tools=[calc_tool],
                budget=BudgetGuard(max_per_day=10.00),
            )
            result = agent.run("What is 15% of 200?")

            assert isinstance(result, Message)
            assert result.content is not None
            assert len(result.content) > 0
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic

    def test_monitoring_events_captured(self):
        """Events should be logged to the monitor store during a run."""
        monitor_store = AgentStore()

        calc_tool = Tool(fn=_calculator, name="calculator", description="Math calculator")
        agent = GovernedAgent(
            name="monitor-test-agent",
            model="gpt-4o-mini",
            tools=[calc_tool],
            budget=BudgetGuard(max_per_day=10.00),
            enable_monitoring=True,
        )

        with patch("agentos.governed_agent.store", monitor_store):
            agent.run("Calculate 10 + 20")

        assert len(monitor_store.events) > 0
        assert "monitor-test-agent" in monitor_store.agents
        agent_data = monitor_store.agents["monitor-test-agent"]
        assert agent_data["total_events"] > 0
        assert agent_data["total_cost"] > 0
        assert agent_data["total_llm_calls"] > 0

    def test_kill_switch_blocks_execution(self):
        """Kill switch should prevent any tool calls."""
        calc_tool = Tool(fn=_calculator, name="calculator", description="Math calculator")
        agent = GovernedAgent(
            name="kill-test-agent",
            model="gpt-4o-mini",
            tools=[calc_tool],
            budget=BudgetGuard(max_per_day=10.00),
        )

        agent.kill("Safety concern")

        check = agent.governance.check_tool_call("calculator", estimated_cost=0.001)
        assert not check.allowed
        assert "kill" in check.rule.lower() or "killed" in check.message.lower()

        agent.revive()
        check_after = agent.governance.check_tool_call("calculator", estimated_cost=0.001)
        assert check_after.allowed

    def test_governed_agent_stats(self):
        """GovernedAgent.get_stats() should return run and cost data."""
        calc_tool = Tool(fn=_calculator, name="calculator", description="Math calculator")
        agent = GovernedAgent(
            name="stats-agent",
            model="gpt-4o-mini",
            tools=[calc_tool],
            budget=BudgetGuard(max_per_day=10.00),
        )

        agent.run("Hello!")

        stats = agent.get_stats()
        assert stats["name"] == "stats-agent"
        assert stats["model"] == "gpt-4o-mini"
        assert "calculator" in stats["tools"]
        assert stats["total_runs"] == 1
        assert stats["total_cost"] > 0
        assert "governance" in stats

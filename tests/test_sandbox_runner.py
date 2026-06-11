from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentos.core.agent import Agent
from agentos.core.types import AgentEvent, Message, Role
from agentos.sandbox.runner import Sandbox
from agentos.sandbox.scenario import Scenario


def _high_scores() -> dict:
    return {
        "relevance": 9.0,
        "quality": 8.0,
        "safety": 10.0,
        "reasoning": "Strong response",
    }


def _low_scores() -> dict:
    return {
        "relevance": 3.0,
        "quality": 2.0,
        "safety": 4.0,
        "reasoning": "Weak response",
    }


class StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _make_agent(response: str = "hello", cost: float = 0.002) -> Agent:
    agent = Agent(name="gate-agent")
    agent.run = MagicMock(return_value=Message(role=Role.ASSISTANT, content=response))
    agent.events = [
        AgentEvent(
            agent_name="gate-agent",
            event_type="llm_call",
            cost_usd=cost,
            latency_ms=10.0,
        )
    ]
    return agent


class TestSandboxPassFailGating:
    def test_passes_when_score_cost_and_latency_within_limits(self):
        agent = _make_agent(cost=0.002)
        scenario = Scenario(
            name="pass-all",
            user_message="hi",
            expected_behavior="greet",
            max_cost=0.05,
            max_latency_ms=5000,
        )
        sandbox = Sandbox(agent, pass_threshold=6.0, embedder=StubEmbedder())

        with patch("agentos.sandbox.runner.judge_response", return_value=_high_scores()):
            with patch("agentos.sandbox.runner.time.time", side_effect=[0.0, 0.05]):
                result = sandbox.run_scenario(scenario)

        assert result.passed is True
        assert result.cost_ok is True
        assert result.latency_ok is True
        assert result.overall_score == pytest.approx(9.0, abs=0.1)

    def test_fails_when_score_below_threshold(self):
        agent = _make_agent(cost=0.001)
        scenario = Scenario(
            name="low-score",
            user_message="hi",
            expected_behavior="greet",
            max_cost=1.0,
            max_latency_ms=10000,
        )
        sandbox = Sandbox(agent, pass_threshold=6.0, embedder=StubEmbedder())

        with patch("agentos.sandbox.runner.judge_response", return_value=_low_scores()):
            with patch("agentos.sandbox.runner.time.time", side_effect=[0.0, 0.01]):
                result = sandbox.run_scenario(scenario)

        assert result.passed is False
        assert result.overall_score < 6.0
        assert result.cost_ok is True
        assert result.latency_ok is True

    def test_fails_when_cost_exceeds_scenario_max(self):
        agent = _make_agent(cost=0.50)
        scenario = Scenario(
            name="cost-breach",
            user_message="hi",
            expected_behavior="greet",
            max_cost=0.10,
            max_latency_ms=10000,
        )
        sandbox = Sandbox(agent, pass_threshold=6.0, embedder=StubEmbedder())

        with patch("agentos.sandbox.runner.judge_response", return_value=_high_scores()):
            with patch("agentos.sandbox.runner.time.time", side_effect=[0.0, 0.01]):
                result = sandbox.run_scenario(scenario)

        assert result.passed is False
        assert result.cost_ok is False
        assert result.latency_ok is True

    def test_fails_when_latency_exceeds_scenario_max(self):
        agent = _make_agent(cost=0.001)
        scenario = Scenario(
            name="slow-run",
            user_message="hi",
            expected_behavior="greet",
            max_cost=1.0,
            max_latency_ms=100,
        )
        sandbox = Sandbox(agent, pass_threshold=6.0, embedder=StubEmbedder())

        with patch("agentos.sandbox.runner.judge_response", return_value=_high_scores()):
            with patch("agentos.sandbox.runner.time.time", side_effect=[0.0, 1.0]):
                result = sandbox.run_scenario(scenario)

        assert result.passed is False
        assert result.latency_ok is False
        assert result.cost_ok is True
        assert result.latency_ms > 100

    def test_fails_when_all_three_gates_breach(self):
        agent = _make_agent(cost=1.0)
        scenario = Scenario(
            name="triple-fail",
            user_message="hi",
            expected_behavior="greet",
            max_cost=0.01,
            max_latency_ms=50,
        )
        sandbox = Sandbox(agent, pass_threshold=8.0, embedder=StubEmbedder())

        with patch("agentos.sandbox.runner.judge_response", return_value=_low_scores()):
            with patch("agentos.sandbox.runner.time.time", side_effect=[0.0, 2.0]):
                result = sandbox.run_scenario(scenario)

        assert result.passed is False
        assert result.cost_ok is False
        assert result.latency_ok is False
        assert result.overall_score < 8.0

    def test_agent_exception_yields_failed_result(self):
        agent = Agent(name="broken")
        agent.run = MagicMock(side_effect=RuntimeError("agent crashed"))
        scenario = Scenario(
            name="error-case",
            user_message="hi",
            expected_behavior="greet",
        )
        sandbox = Sandbox(agent, embedder=StubEmbedder())
        result = sandbox.run_scenario(scenario)
        assert result.passed is False
        assert result.error == "agent crashed"

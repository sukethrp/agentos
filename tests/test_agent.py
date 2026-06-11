from __future__ import annotations

from unittest.mock import patch

import pytest

from agentos.core.agent import Agent, _cache_key, _tool_cache
from agentos.core.tool import Tool
from agentos.core.types import AgentEvent, Message, Role, ToolCall, ToolExecutionContext


def _llm_event(cost: float = 0.001) -> AgentEvent:
    return AgentEvent(
        agent_name="test",
        event_type="llm_call",
        tokens_used=10,
        cost_usd=cost,
        latency_ms=1.0,
    )


def _final_message(content: str = "done") -> Message:
    return Message(role=Role.ASSISTANT, content=content)


def _tool_message(tool_name: str, args: dict | None = None) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=[
            ToolCall(name=tool_name, arguments=args or {}),
        ],
    )


@pytest.fixture(autouse=True)
def _clear_tool_cache():
    _tool_cache.clear()
    yield
    _tool_cache.clear()


class TestAgentToolCallingLoop:
    def test_executes_tools_then_returns_final_answer(self):
        def echo(message: str) -> str:
            return message.upper()

        tool = Tool(echo, name="echo")
        agent = Agent(name="loop-agent", tools=[tool], max_iterations=5)
        calls = [
            (_tool_message("echo", {"message": "hi"}), _llm_event()),
            (_final_message("ECHO: HI"), _llm_event()),
        ]

        with patch("agentos.core.agent.call_llm", side_effect=calls):
            result = agent.run("say hi")

        assert result.content == "ECHO: HI"
        assert len(agent.events) == 3
        tool_events = [e for e in agent.events if e.event_type == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0].data["tool"] == "echo"
        assert tool_events[0].data["result"] == "HI"

    def test_appends_tool_results_to_messages(self):
        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(add, name="add")
        agent = Agent(tools=[tool], max_iterations=3)
        calls = [
            (_tool_message("add", {"a": 2, "b": 3}), _llm_event()),
            (_final_message("5"), _llm_event()),
        ]

        with patch("agentos.core.agent.call_llm", side_effect=calls):
            agent.run("add 2 and 3")

        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "5"


class TestAgentMaxIterations:
    def test_returns_fallback_when_iteration_cap_hit(self):
        agent = Agent(max_iterations=2)
        perpetual = (_tool_message("missing"), _llm_event())

        with patch("agentos.core.agent.call_llm", return_value=perpetual):
            result = agent.run("never finish")

        assert result.content == "Could not complete the task."
        assert len([e for e in agent.events if e.event_type == "llm_call"]) == 2


class TestAgentErrorHandling:
    def test_unknown_tool_produces_error_string(self):
        agent = Agent(tools=[], max_iterations=1)
        calls = [(_tool_message("ghost"), _llm_event())]

        with patch("agentos.core.agent.call_llm", side_effect=calls):
            agent.run("call ghost")

        tool_events = [e for e in agent.events if e.event_type == "tool_call"]
        assert tool_events[0].data["result"].startswith("ERROR: Tool 'ghost' not found")

    def test_tool_exception_is_caught_in_execute_tool_with_retry(self):
        def boom() -> None:
            raise ValueError("boom")

        tool = Tool(boom, name="boom")
        agent = Agent(tools=[tool])
        tc = ToolCall(name="boom", arguments={})
        result = agent._execute_tool_with_retry(tool, tc)
        assert result.result.startswith("ERROR:")
        assert "boom" in result.result

    def test_execute_tools_batch_handles_missing_tool(self):
        agent = Agent(tools=[])
        ctx = ToolExecutionContext(agent_id="a", session_id="s")
        results = agent._execute_tools_batch(
            [ToolCall(name="nope", arguments={})], ctx
        )
        assert results[0][0].startswith("ERROR: Tool 'nope' not found")

    def test_tool_cache_returns_cached_result(self):
        def stable(x: str) -> str:
            return f"out:{x}"

        tool = Tool(stable, name="stable")
        agent = Agent(tools=[tool])
        ctx = ToolExecutionContext(agent_id="a", session_id="s")
        tc = ToolCall(name="stable", arguments={"x": "1"})
        first = agent._execute_tools_batch([tc], ctx)
        second = agent._execute_tools_batch([tc], ctx)

        assert first[0][0] == "out:1"
        assert second[0][0] == "out:1"
        assert second[0][1] == 0.0


class TestAgentCacheKey:
    def test_cache_key_is_stable_for_arg_order(self):
        k1 = _cache_key("t", {"a": 1, "b": 2})
        k2 = _cache_key("t", {"b": 2, "a": 1})
        assert k1 == k2

    def test_cache_key_differs_for_different_tools(self):
        assert _cache_key("a", {}) != _cache_key("b", {})


class TestAgentDelegate:
    def test_delegate_returns_delegate_response_content(self):
        parent = Agent(name="parent")
        child = Agent(name="child")

        with patch.object(child, "run", return_value=_final_message("child answer")):
            assert parent.delegate(child, "subtask") == "child answer"

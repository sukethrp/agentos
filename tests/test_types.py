import re

from agentos.core.types import (
    Role,
    ToolParam,
    ToolSpec,
    ToolCall,
    ToolResult,
    Message,
    AgentConfig,
    AgentEvent,
)


def test_toolspec_to_openai_schema_basic():
    params = [
        ToolParam(name="x", type="number", description="x value"),
        ToolParam(name="y", type="number", description="y value", required=False),
    ]
    spec = ToolSpec(name="add", description="Add two numbers", parameters=params)

    schema = spec.to_openai_schema()

    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "add"
    assert func["description"] == "Add two numbers"
    params_obj = func["parameters"]
    assert params_obj["type"] == "object"
    props = params_obj["properties"]
    assert set(props.keys()) == {"x", "y"}
    assert props["x"]["type"] == "number"
    assert props["y"]["type"] == "number"
    required = set(params_obj["required"])
    assert "x" in required
    assert "y" not in required


def test_message_toolcall_agent_event_creation():
    call = ToolCall(name="calc", arguments={"expression": "1+2"})
    assert call.id.startswith("call_")
    assert call.arguments["expression"] == "1+2"

    msg = Message(role=Role.USER, content="hi", tool_calls=[call])
    assert msg.role is Role.USER
    assert msg.content == "hi"
    assert msg.tool_calls and msg.tool_calls[0].name == "calc"

    cfg = AgentConfig(name="test-agent", model="gpt-4o-mini")
    assert cfg.name == "test-agent"
    assert cfg.max_tokens > 0

    evt = AgentEvent(agent_name="test-agent", event_type="llm_call", tokens_used=10, cost_usd=0.001)
    assert evt.agent_name == "test-agent"
    assert evt.event_type == "llm_call"
    assert evt.tokens_used == 10
    assert evt.cost_usd == 0.001
    assert evt.timestamp > 0

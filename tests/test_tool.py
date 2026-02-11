from agentos.core.tool import Tool, tool
from agentos.core.types import ToolCall


def test_tool_decorator_creates_tool():
    @tool(description="Add two numbers")
    def add(x: int, y: int) -> int:
        return x + y

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert "Add two numbers" in add.description


def test_tool_param_inference_types_and_required():
    def fn(a: int, flag: bool, text: str = "hi") -> str:
        return text

    t = Tool(fn)
    params = {p.name: p for p in t.params}

    assert params["a"].type == "number"
    assert params["a"].required is True
    assert params["flag"].type == "boolean"
    assert params["text"].type == "string"
    assert params["text"].required is False


def test_tool_execute_success_and_latency():
    def echo(message: str) -> str:
        return message.upper()

    t = Tool(echo)
    call = ToolCall(name=t.name, arguments={"message": "hello"})

    result = t.execute(call)
    assert result.name == t.name
    assert result.tool_call_id == call.id
    assert result.result == "HELLO"
    assert result.latency_ms >= 0


def test_tool_execute_error_handling():
    def boom() -> None:
        raise ValueError("boom")

    t = Tool(boom)
    call = ToolCall(name=t.name, arguments={})

    result = t.execute(call)
    assert result.name == t.name
    assert result.tool_call_id == call.id
    assert result.result.startswith("ERROR:")
    assert "boom" in result.result

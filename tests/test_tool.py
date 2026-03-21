"""Tests for the Tool class and @tool decorator.

Covers:
- @tool decorator: name extraction, description, parameter inference
- Tool with no parameters
- Tool with optional parameters (default values)
- Tool that raises various exceptions
- Tool with complex return types (dict, list, nested objects)
- Tool.spec property (OpenAI-compatible schema)
- Custom name/description overrides
"""

from agentos.core.tool import Tool, tool
from agentos.core.types import ToolCall


# ═══════════════════════════════════════════════════════════════════
# @tool decorator
# ═══════════════════════════════════════════════════════════════════


class TestToolDecoratorExtraction:
    """Test that @tool correctly extracts metadata from the function."""

    def test_extracts_function_name(self):
        """Decorator should use the function's __name__ when no name is given."""
        @tool(description="Add two numbers")
        def add(x: int, y: int) -> int:
            return x + y

        assert isinstance(add, Tool)
        assert add.name == "add"

    def test_extracts_description(self):
        """Decorator should use the provided description."""
        @tool(description="Add two numbers")
        def add(x: int, y: int) -> int:
            return x + y

        assert add.description == "Add two numbers"

    def test_extracts_parameters_from_signature(self):
        """Decorator should infer parameter names, types, and required status."""
        @tool(description="Concatenate strings")
        def concat(a: str, b: str, sep: str = " ") -> str:
            return f"{a}{sep}{b}"

        params = {p.name: p for p in concat.params}
        assert len(params) == 3
        assert params["a"].type == "string"
        assert params["a"].required is True
        assert params["b"].required is True
        assert params["sep"].required is False

    def test_custom_name_overrides_function_name(self):
        """Explicit name= should override the function's __name__."""
        @tool(name="my_adder", description="Adds")
        def add(x: int, y: int) -> int:
            return x + y

        assert add.name == "my_adder"

    def test_falls_back_to_docstring_when_no_description(self):
        """When no description is given to Tool(), it should fall back to the docstring."""
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        t = Tool(multiply)
        assert t.description == "Multiply two integers."

    def test_falls_back_to_generic_when_no_docstring(self):
        """When there's no docstring or description, use a generic fallback."""
        def mystery(x: str) -> str:
            return x

        t = Tool(mystery)
        assert "mystery" in t.description

    def test_decorator_preserves_timeout_and_retries(self):
        """Custom timeout and retry settings should be stored on the Tool."""
        @tool(description="Slow op", timeout_seconds=60.0, max_retries=3)
        def slow_op() -> str:
            return "done"

        assert slow_op.timeout_seconds == 60.0
        assert slow_op.max_retries == 3


# ═══════════════════════════════════════════════════════════════════
# Parameter inference
# ═══════════════════════════════════════════════════════════════════


class TestToolParamInference:
    """Test that Tool._infer_params correctly maps Python types."""

    def test_infers_int_as_number(self):
        """int annotations should map to JSON schema 'number'."""
        def fn(a: int) -> int:
            return a
        t = Tool(fn)
        assert t.params[0].type == "number"

    def test_infers_float_as_number(self):
        """float annotations should also map to 'number'."""
        def fn(a: float) -> float:
            return a
        t = Tool(fn)
        assert t.params[0].type == "number"

    def test_infers_bool_as_boolean(self):
        """bool annotations should map to 'boolean'."""
        def fn(flag: bool) -> bool:
            return flag
        t = Tool(fn)
        assert t.params[0].type == "boolean"

    def test_infers_str_as_string(self):
        """str annotations should map to 'string'."""
        def fn(text: str) -> str:
            return text
        t = Tool(fn)
        assert t.params[0].type == "string"

    def test_unannotated_param_defaults_to_string(self):
        """Parameters without type annotations should default to 'string'."""
        def fn(x) -> str:
            return str(x)
        t = Tool(fn)
        assert t.params[0].type == "string"

    def test_required_vs_optional(self):
        """Params with defaults are optional; those without are required."""
        def fn(a: int, flag: bool, text: str = "hi") -> str:
            return text

        t = Tool(fn)
        params = {p.name: p for p in t.params}

        assert params["a"].required is True
        assert params["flag"].required is True
        assert params["text"].required is False


# ═══════════════════════════════════════════════════════════════════
# Tool with no parameters
# ═══════════════════════════════════════════════════════════════════


class TestToolNoParameters:
    """Test tools that take zero arguments."""

    def test_no_param_tool_has_empty_params_list(self, no_param_tool):
        """A zero-arg function should produce an empty params list."""
        assert no_param_tool.params == []

    def test_no_param_tool_executes_correctly(self, no_param_tool):
        """Executing a zero-arg tool should return its value."""
        call = ToolCall(name=no_param_tool.name, arguments={})
        result = no_param_tool.execute(call)
        assert result.result == "2025-01-01T00:00:00Z"

    def test_no_param_tool_spec_has_empty_parameters(self, no_param_tool):
        """The spec for a zero-arg tool should list no parameters."""
        spec = no_param_tool.spec
        assert spec.parameters == []


# ═══════════════════════════════════════════════════════════════════
# Tool with optional parameters (default values)
# ═══════════════════════════════════════════════════════════════════


class TestToolOptionalParameters:
    """Test tools where some arguments have default values."""

    def test_optional_param_not_required(self):
        """Parameters with default values should be marked required=False."""
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"
        t = Tool(greet)
        params = {p.name: p for p in t.params}
        assert params["name"].required is True
        assert params["greeting"].required is False

    def test_execution_with_defaults_omitted(self):
        """Tool should work when optional args are not provided."""
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"
        t = Tool(greet)
        call = ToolCall(name=t.name, arguments={"name": "World"})
        result = t.execute(call)
        assert result.result == "Hello, World!"

    def test_execution_with_defaults_overridden(self):
        """Optional args should be overridable."""
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"
        t = Tool(greet)
        call = ToolCall(name=t.name, arguments={"name": "World", "greeting": "Hi"})
        result = t.execute(call)
        assert result.result == "Hi, World!"

    def test_all_optional_params(self):
        """A tool where every parameter is optional."""
        def configure(timeout: int = 30, verbose: bool = False) -> str:
            return f"timeout={timeout}, verbose={verbose}"
        t = Tool(configure)
        assert all(not p.required for p in t.params)
        call = ToolCall(name=t.name, arguments={})
        result = t.execute(call)
        assert "timeout=30" in result.result


# ═══════════════════════════════════════════════════════════════════
# Tool that raises exceptions
# ═══════════════════════════════════════════════════════════════════


class TestToolExceptionHandling:
    """Test that tool execution errors are caught and returned as ERROR strings."""

    def test_value_error_is_caught(self):
        """ValueError should be caught and returned as an ERROR result."""
        def boom() -> None:
            raise ValueError("boom")

        t = Tool(boom)
        call = ToolCall(name=t.name, arguments={})
        result = t.execute(call)
        assert result.result.startswith("ERROR:")
        assert "boom" in result.result

    def test_type_error_is_caught(self):
        """TypeError from bad arguments should be caught gracefully."""
        def strict(x: int) -> int:
            return x + 1

        t = Tool(strict)
        call = ToolCall(name=t.name, arguments={"x": "not_a_number"})
        result = t.execute(call)
        assert result.result.startswith("ERROR:")

    def test_runtime_error_is_caught(self):
        """RuntimeError should be caught like any other exception."""
        def fragile() -> str:
            raise RuntimeError("something broke")

        t = Tool(fragile)
        call = ToolCall(name=t.name, arguments={})
        result = t.execute(call)
        assert "something broke" in result.result

    def test_error_result_still_has_latency(self):
        """Even failed executions should record latency."""
        def fail() -> None:
            raise Exception("fail")

        t = Tool(fail)
        call = ToolCall(name=t.name, arguments={})
        result = t.execute(call)
        assert result.latency_ms >= 0

    def test_error_result_preserves_tool_call_id(self):
        """The tool_call_id should be preserved even on error."""
        def fail() -> None:
            raise Exception("fail")

        t = Tool(fail)
        call = ToolCall(name=t.name, arguments={})
        result = t.execute(call)
        assert result.tool_call_id == call.id
        assert result.name == t.name


# ═══════════════════════════════════════════════════════════════════
# Tool with complex return types
# ═══════════════════════════════════════════════════════════════════


class TestToolComplexReturnTypes:
    """Test that tools returning non-string types are str()-ified."""

    def test_dict_return_is_stringified(self):
        """A tool returning a dict should have it converted to a string."""
        def lookup(key: str) -> dict:
            return {"key": key, "value": 42}

        t = Tool(lookup)
        call = ToolCall(name=t.name, arguments={"key": "answer"})
        result = t.execute(call)
        assert "'key': 'answer'" in result.result
        assert "'value': 42" in result.result

    def test_list_return_is_stringified(self):
        """A tool returning a list should have it converted to a string."""
        def get_items() -> list:
            return ["a", "b", "c"]

        t = Tool(get_items)
        call = ToolCall(name=t.name, arguments={})
        result = t.execute(call)
        assert "['a', 'b', 'c']" in result.result

    def test_int_return_is_stringified(self):
        """A tool returning an int should have it stringified."""
        def compute(x: int, y: int) -> int:
            return x * y

        t = Tool(compute)
        call = ToolCall(name=t.name, arguments={"x": 6, "y": 7})
        result = t.execute(call)
        assert result.result == "42"

    def test_none_return_is_stringified(self):
        """A tool returning None should produce the string 'None'."""
        def noop() -> None:
            pass

        t = Tool(noop)
        call = ToolCall(name=t.name, arguments={})
        result = t.execute(call)
        assert result.result == "None"

    def test_nested_dict_return(self):
        """A tool returning deeply nested data should stringify cleanly."""
        def deep() -> dict:
            return {"outer": {"inner": [1, 2, 3]}}

        t = Tool(deep)
        call = ToolCall(name=t.name, arguments={})
        result = t.execute(call)
        assert "inner" in result.result
        assert "[1, 2, 3]" in result.result


# ═══════════════════════════════════════════════════════════════════
# Tool.spec property
# ═══════════════════════════════════════════════════════════════════


class TestToolSpec:
    """Test the OpenAI-compatible tool spec output."""

    def test_spec_contains_name_and_description(self, simple_tool):
        """Spec should carry the tool's name and description."""
        spec = simple_tool.spec
        assert spec.name == "echo"
        assert spec.description is not None

    def test_spec_parameters_match_function_signature(self):
        """Spec parameters should reflect the function's typed signature."""
        def search(query: str, limit: int = 10) -> str:
            return query

        t = Tool(search, description="Search things")
        spec = t.spec
        param_names = [p.name for p in spec.parameters]
        assert "query" in param_names
        assert "limit" in param_names


# ═══════════════════════════════════════════════════════════════════
# Execution: success path
# ═══════════════════════════════════════════════════════════════════


class TestToolExecutionSuccess:
    """Test the happy path for tool execution."""

    def test_execute_returns_correct_result(self, simple_tool):
        """Executing a tool with valid args should return the function's output."""
        call = ToolCall(name=simple_tool.name, arguments={"message": "hello"})
        result = simple_tool.execute(call)
        assert result.result == "HELLO"

    def test_execute_records_latency(self, simple_tool):
        """Execution should measure wall-clock latency in milliseconds."""
        call = ToolCall(name=simple_tool.name, arguments={"message": "test"})
        result = simple_tool.execute(call)
        assert result.latency_ms >= 0

    def test_execute_preserves_tool_call_id(self, simple_tool):
        """The result should carry forward the original ToolCall's id."""
        call = ToolCall(name=simple_tool.name, arguments={"message": "test"})
        result = simple_tool.execute(call)
        assert result.tool_call_id == call.id
        assert result.name == simple_tool.name

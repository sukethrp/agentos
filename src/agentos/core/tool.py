"""``@tool`` decorator that introspects function signatures for LLM function calling.

This module provides two ways to create tools that an :class:`Agent` can invoke:

1. **Imperative** — instantiate :class:`Tool` directly::

       Tool(fn=my_func, name="my_tool", description="Does X")

2. **Declarative** — use the :func:`tool` decorator::

       @tool(description="Does X")
       def my_tool(query: str) -> str: ...

In both cases the constructor inspects the wrapped function's type hints
via :mod:`inspect` and builds a list of :class:`ToolParam` objects.  These
are later serialised to **JSON Schema** (by the provider layer) so the LLM
knows what arguments the tool accepts and can generate valid calls.

**Parameter type mapping:**

=============  =============
Python hint    JSON Schema
=============  =============
``str``        ``"string"``
``int/float``  ``"number"``
``bool``       ``"boolean"``
*anything else*  ``"string"``
=============  =============

Parameters without a default value are marked ``required``.  This simple
heuristic covers the vast majority of tools; for complex schemas (nested
objects, enums) authors can override ``Tool.params`` after construction.
"""

from __future__ import annotations
import time
import inspect
from typing import Callable
from agentos.core.types import ToolSpec, ToolParam, ToolCall, ToolResult


class Tool:
    """A callable tool that an LLM agent can invoke during the ReAct loop.

    The ``Tool`` wraps an ordinary Python function and attaches the metadata
    (name, description, parameter schema) that LLM providers need to advertise
    the tool via function-calling APIs.

    Args:
        fn: The Python callable to execute when the tool is invoked.
        name: Tool name exposed to the LLM.  Defaults to ``fn.__name__``.
        description: Human-readable description sent to the LLM so it knows
            when to use this tool.  Falls back to the function's docstring.
        timeout_seconds: Maximum wall-clock time for a single invocation
            before it is cancelled with a ``TimeoutError``.
        max_retries: Number of automatic retries on failure (with exponential
            back-off).  ``0`` means no retries.

    Example::

        from agentos.core.tool import Tool

        def get_weather(city: str) -> str:
            return f"Weather in {city}: 72°F, sunny"

        weather = Tool(fn=get_weather, description="Get current weather")
        spec = weather.spec          # ToolSpec for the provider
        result = weather.execute(call)  # ToolResult with latency
    """

    def __init__(
        self,
        fn: Callable[..., str],
        name: str | None = None,
        description: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or fn.__doc__ or f"Tool: {self.name}"
        self.params = self._infer_params()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _infer_params(self) -> list[ToolParam]:
        """Extract parameter metadata from the function's type hints.

        Iterates over ``inspect.signature(self.fn).parameters`` and maps each
        annotation to a JSON-Schema-compatible type string.  Parameters with
        no default value are marked as ``required`` so the LLM knows it must
        supply them.

        Returns:
            Ordered list of :class:`ToolParam` objects, one per function arg.
        """
        params = []
        sig = inspect.signature(self.fn)
        for pname, p in sig.parameters.items():
            hint = p.annotation
            # Map Python types to JSON Schema type strings.  We intentionally
            # fall back to "string" for unknown types because most LLMs handle
            # string coercion gracefully, and it avoids crashing on custom types.
            ptype = "string"
            if hint is int or hint is float:
                ptype = "number"
            elif hint is bool:
                ptype = "boolean"
            params.append(
                ToolParam(
                    name=pname,
                    type=ptype,
                    description=f"Parameter: {pname}",
                    required=p.default == inspect.Parameter.empty,
                )
            )
        return params

    @property
    def spec(self) -> ToolSpec:
        """Return the JSON-Schema-ready specification for this tool.

        The returned :class:`ToolSpec` is what provider adapters (OpenAI,
        Anthropic, etc.) convert into their respective function-calling
        payloads.

        Returns:
            A :class:`ToolSpec` containing name, description, and parameter
            definitions.
        """
        return ToolSpec(
            name=self.name, description=self.description, parameters=self.params
        )

    def execute(self, call: ToolCall) -> ToolResult:
        """Invoke the underlying function and return a :class:`ToolResult`.

        The function is called with ``**call.arguments`` so argument names
        must match the function signature.  Exceptions are caught and
        returned as ``"ERROR: ..."`` strings rather than propagated, keeping
        the agent loop alive even when a tool misbehaves.

        Args:
            call: The :class:`ToolCall` containing the function name and
                keyword arguments selected by the LLM.

        Returns:
            A :class:`ToolResult` with the stringified return value (or error
            message) and wall-clock latency in milliseconds.

        Raises:
            No exceptions are raised; errors are captured in the result.
        """
        start = time.time()
        try:
            result = self.fn(**call.arguments)
            latency = (time.time() - start) * 1000
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                result=str(result),
                latency_ms=round(latency, 2),
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                result=f"ERROR: {e}",
                latency_ms=round(latency, 2),
            )


def tool(
    name: str | None = None,
    description: str | None = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 0,
) -> Callable[[Callable[..., str]], Tool]:
    """Decorator that converts a plain function into a :class:`Tool`.

    This is syntactic sugar for ``Tool(fn=..., name=..., description=...)``.
    The decorated object is replaced by a :class:`Tool` instance, so it
    can be passed directly into ``Agent(tools=[...])``.

    Args:
        name: Tool name exposed to the LLM.  Defaults to the function name.
        description: Human-readable description for the LLM.
        timeout_seconds: Per-invocation timeout.
        max_retries: Automatic retry count on failure.

    Returns:
        A decorator that accepts a callable and returns a :class:`Tool`.

    Example::

        @tool(description="Calculate a math expression safely")
        def calculator(expression: str) -> str:
            return str(eval(expression))  # simplified

        # `calculator` is now a Tool instance:
        agent = Agent(tools=[calculator])
    """
    def decorator(fn: Callable[..., str]) -> Tool:
        return Tool(
            fn=fn,
            name=name or fn.__name__,
            description=description,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    return decorator

from __future__ import annotations
import time
import inspect
from typing import Any, Callable
from agentos.core.types import ToolSpec, ToolParam, ToolCall, ToolResult


class Tool:
    def __init__(self, fn: Callable, name: str | None = None, description: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or fn.__doc__ or f"Tool: {self.name}"
        self.params = self._infer_params()

    def _infer_params(self) -> list[ToolParam]:
        params = []
        sig = inspect.signature(self.fn)
        for pname, p in sig.parameters.items():
            hint = p.annotation
            ptype = "string"
            if hint == int or hint == float:
                ptype = "number"
            elif hint == bool:
                ptype = "boolean"
            params.append(ToolParam(
                name=pname,
                type=ptype,
                description=f"Parameter: {pname}",
                required=p.default == inspect.Parameter.empty,
            ))
        return params

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.params)

    def execute(self, call: ToolCall) -> ToolResult:
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


def tool(name: str | None = None, description: str | None = None):
    def decorator(fn: Callable) -> Tool:
        return Tool(fn=fn, name=name or fn.__name__, description=description)
    return decorator
"""Conversions between AgentOS tool types and MCP protocol types."""

from __future__ import annotations

from agentos.core.types import ToolSpec


def toolspec_to_input_schema(spec: ToolSpec) -> dict:
    """Convert an AgentOS ToolSpec to an MCP-compatible JSON Schema dict."""
    properties: dict[str, dict] = {}
    required: list[str] = []

    for param in spec.parameters:
        prop: dict = {"type": param.type, "description": param.description}
        if param.enum:
            prop["enum"] = param.enum
        properties[param.name] = prop
        if param.required:
            required.append(param.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }

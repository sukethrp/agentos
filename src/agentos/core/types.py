from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum
import time
import uuid


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolParam(BaseModel):
    name: str
    type: str = "string"
    description: str
    required: bool = True
    enum: list[str] | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: list[ToolParam] = Field(default_factory=list)

    def to_openai_schema(self) -> dict:
        props = {}
        req = []
        for p in self.parameters:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.enum:
                props[p.name]["enum"] = p.enum
            if p.required:
                req.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": req,
                },
            },
        }


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    result: str
    latency_ms: float = 0.0


class ToolExecutionContext(BaseModel):
    agent_id: str = ""
    session_id: str = ""
    budget_remaining: float = 0.0


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class AgentConfig(BaseModel):
    name: str = "agent"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    max_iterations: int = 10
    max_tokens: int = 1024
    temperature: float = 0.7


class AgentEvent(BaseModel):
    agent_name: str
    event_type: str
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

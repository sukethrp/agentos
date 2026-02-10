"""Ollama provider for AgentOS — run models locally for FREE."""

from __future__ import annotations
import time
import json
import httpx
from agentos.core.types import Message, ToolCall, AgentEvent, Role
from agentos.core.tool import Tool

OLLAMA_URL = "http://localhost:11434"


def _tools_to_ollama_schema(tools: list[Tool]) -> list[dict]:
    """Convert AgentOS tools to Ollama tool format."""
    schemas = []
    for t in tools:
        props = {}
        req = []
        for p in t.params:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                req.append(p.name)
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": req,
                },
            },
        })
    return schemas


def call_ollama(
    messages: list[dict],
    tools: list[Tool],
    model: str = "llama3.1",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    """Make a call to local Ollama server."""

    # Convert messages — strip tool-specific fields for basic Ollama support
    ollama_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            ollama_messages.append({"role": "user", "content": f"Tool result: {content}"})
        elif role == "assistant" and msg.get("tool_calls"):
            # Skip assistant tool-call messages, Ollama handles differently
            if content:
                ollama_messages.append({"role": "assistant", "content": content})
        else:
            ollama_messages.append({"role": role, "content": content or ""})

    tool_schemas = _tools_to_ollama_schema(tools) if tools else []

    start = time.time()

    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if tool_schemas:
        payload["tools"] = tool_schemas

    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        raise ConnectionError(
            "Cannot connect to Ollama. Make sure Ollama is running: 'ollama serve'"
        )

    latency = (time.time() - start) * 1000

    # Parse response
    message_data = data.get("message", {})
    content = message_data.get("content", "")

    # Parse tool calls if present
    parsed_tool_calls = None
    raw_tool_calls = message_data.get("tool_calls", [])
    if raw_tool_calls:
        parsed_tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            parsed_tool_calls.append(ToolCall(
                name=fn.get("name", ""),
                arguments=fn.get("arguments", {}),
            ))

    # Estimate tokens (Ollama doesn't always return exact counts)
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    msg = Message(
        role=Role.ASSISTANT,
        content=content or None,
        tool_calls=parsed_tool_calls,
    )

    event = AgentEvent(
        agent_name=agent_name,
        event_type="llm_call",
        data={
            "model": model,
            "provider": "ollama",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "has_tool_calls": parsed_tool_calls is not None,
        },
        tokens_used=prompt_tokens + completion_tokens,
        cost_usd=0.0,  # Local models are free!
        latency_ms=round(latency, 2),
    )

    return msg, event
"""Anthropic Claude provider for AgentOS."""

from __future__ import annotations
import time
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from agentos.core.types import Message, ToolCall, AgentEvent, Role
from agentos.core.tool import Tool

load_dotenv()

PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-sonnet": {"input": 3.00, "output": 15.00},
    "claude-haiku": {"input": 0.25, "output": 1.25},
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    "claude-opus": {"input": 15.00, "output": 75.00},
}


def _tools_to_anthropic_schema(tools: list[Tool]) -> list[dict]:
    """Convert AgentOS tools to Anthropic tool format."""
    schemas = []
    for t in tools:
        props = {}
        req = []
        for p in t.params:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                req.append(p.name)
        schemas.append({
            "name": t.name,
            "description": t.description,
            "input_schema": {
                "type": "object",
                "properties": props,
                "required": req,
            },
        })
    return schemas


def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-style messages to Anthropic format.
    Returns (system_prompt, messages_list).
    """
    system = ""
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            system = msg.get("content", "")

        elif role == "user":
            anthropic_messages.append({"role": "user", "content": msg.get("content", "")})

        elif role == "assistant":
            content_blocks = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args.replace("'", '"'))
                        except json.JSONDecodeError:
                            args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
            if content_blocks:
                anthropic_messages.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }],
            })

    return system, anthropic_messages


def call_anthropic(
    messages: list[dict],
    tools: list[Tool],
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    """Make a single call to Anthropic Claude."""
    client = Anthropic()

    # Map short names to full model IDs
    model_map = {
        "claude-sonnet": "claude-sonnet-4-20250514",
        "claude-haiku": "claude-haiku-4-5-20251001",
        "claude-opus": "claude-opus-4-6",
    }
    actual_model = model_map.get(model, model)

    system_prompt, anthropic_msgs = _convert_messages(messages)
    tool_schemas = _tools_to_anthropic_schema(tools) if tools else []

    start = time.time()

    kwargs = {
        "model": actual_model,
        "max_tokens": max_tokens,
        "messages": anthropic_msgs,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if tool_schemas:
        kwargs["tools"] = tool_schemas
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.messages.create(**kwargs)
    latency = (time.time() - start) * 1000

    # Parse response
    content_text = ""
    parsed_tool_calls = None

    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            if parsed_tool_calls is None:
                parsed_tool_calls = []
            parsed_tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input if isinstance(block.input, dict) else {},
            ))

    # Calculate cost
    prices = PRICING.get(model, {"input": 3.00, "output": 15.00})
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000

    msg = Message(
        role=Role.ASSISTANT,
        content=content_text or None,
        tool_calls=parsed_tool_calls,
    )

    event = AgentEvent(
        agent_name=agent_name,
        event_type="llm_call",
        data={
            "model": actual_model,
            "provider": "anthropic",
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "has_tool_calls": parsed_tool_calls is not None,
        },
        tokens_used=input_tokens + output_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency, 2),
    )

    return msg, event
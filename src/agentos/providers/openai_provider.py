from __future__ import annotations
import time
import json
from typing import Generator
from openai import OpenAI
from dotenv import load_dotenv
from agentos.core.types import Message, ToolCall, AgentEvent, Role
from agentos.core.tool import Tool

load_dotenv()

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def call_llm_stream(
    messages: list[dict],
    tools: list[Tool],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> Generator[str | tuple[str, Message, AgentEvent], None, None]:
    """Stream LLM response, yielding tokens. Yields ("done", msg, event) or ("tool_calls", msg, event) at end."""
    client = OpenAI()
    tool_schemas = [t.spec.to_openai_schema() for t in tools] if tools else None

    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tool_schemas,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    content_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if getattr(delta, "content", None):
            content_parts.append(delta.content)
            yield delta.content

        if getattr(delta, "tool_calls", None) and delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index if tc.index is not None else 0
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                acc = tool_calls_acc[idx]
                if tc.id:
                    acc["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function.arguments:
                        acc["arguments"] += tc.function.arguments or ""

    latency = (time.time() - start) * 1000
    full_content = "".join(content_parts)

    parsed_tool_calls = None
    if tool_calls_acc:
        parsed_tool_calls = []
        for idx in sorted(tool_calls_acc.keys()):
            acc = tool_calls_acc[idx]
            if acc["name"]:
                try:
                    args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_calls.append(
                    ToolCall(
                        id=acc["id"] or f"call_{idx}",
                        name=acc["name"],
                        arguments=args,
                    )
                )

    prices = PRICING.get(model, {"input": 0.15, "output": 0.60})
    completion_tokens = max(1, len(full_content) // 4)
    cost = (0 + completion_tokens * prices["output"]) / 1_000_000

    msg = Message(
        role=Role.ASSISTANT,
        content=full_content or None,
        tool_calls=parsed_tool_calls,
    )

    event = AgentEvent(
        agent_name=agent_name,
        event_type="llm_call",
        data={
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "has_tool_calls": parsed_tool_calls is not None,
        },
        tokens_used=completion_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency, 2),
    )

    if parsed_tool_calls:
        yield ("tool_calls", msg, event)
    else:
        yield ("done", msg, event)


def call_llm(
    messages: list[dict],
    tools: list[Tool],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    client = OpenAI()

    tool_schemas = [t.spec.to_openai_schema() for t in tools] if tools else None

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tool_schemas,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = (time.time() - start) * 1000

    choice = response.choices[0].message
    usage = response.usage

    prices = PRICING.get(model, {"input": 0.15, "output": 0.60})
    cost = (
        usage.prompt_tokens * prices["input"]
        + usage.completion_tokens * prices["output"]
    ) / 1_000_000

    parsed_tool_calls = None
    if choice.tool_calls:
        parsed_tool_calls = []
        for tc in choice.tool_calls:
            parsed_tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
            )

    msg = Message(
        role=Role.ASSISTANT,
        content=choice.content,
        tool_calls=parsed_tool_calls,
    )

    event = AgentEvent(
        agent_name=agent_name,
        event_type="llm_call",
        data={
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "has_tool_calls": parsed_tool_calls is not None,
        },
        tokens_used=usage.total_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency, 2),
    )

    return msg, event

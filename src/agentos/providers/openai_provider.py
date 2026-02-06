from __future__ import annotations
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
from agentos.core.types import Message, ToolCall, AgentEvent, Role
from agentos.core.tool import Tool

load_dotenv()

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


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
    cost = (usage.prompt_tokens * prices["input"] + usage.completion_tokens * prices["output"]) / 1_000_000

    parsed_tool_calls = None
    if choice.tool_calls:
        parsed_tool_calls = []
        for tc in choice.tool_calls:
            parsed_tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            ))

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
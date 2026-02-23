from __future__ import annotations
import asyncio
import json
import time
from typing import AsyncGenerator
from anthropic import AsyncAnthropic, RateLimitError
from dotenv import load_dotenv
from agentos.core.types import Message, ToolCall, AgentEvent, Role
from agentos.core.tool import Tool
from agentos.providers.base import BaseProvider

load_dotenv()

PRICING = {
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 0.25, "output": 1.25},
}

MODELS = {"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"}


def _tools_to_anthropic_schema(tools: list[Tool]) -> list[dict]:
    schemas = []
    for t in tools:
        props = {}
        req = []
        for p in t.params:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                req.append(p.name)
        schemas.append(
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": props,
                    "required": req,
                },
            }
        )
    return schemas


def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system = ""
    anthropic_messages = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            system = msg.get("content", "")
        elif role == "user":
            anthropic_messages.append(
                {"role": "user", "content": msg.get("content", "")}
            )
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
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        }
                    )
            if content_blocks:
                anthropic_messages.append(
                    {"role": "assistant", "content": content_blocks}
                )
        elif role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }
                    ],
                }
            )
    return system, anthropic_messages


def _parse_content_blocks(content: list) -> tuple[str, list[ToolCall] | None]:
    text_parts = []
    tool_calls = None
    for block in content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            if tool_calls is None:
                tool_calls = []
            args = block.input if isinstance(block.input, dict) else {}
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=args))
    return "".join(text_parts), tool_calls


async def _with_retry(coro_factory, max_retries: int = 5):
    base_delay = 1.0
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except RateLimitError as e:
            last_error = e
            if attempt == max_retries - 1:
                raise
            retry_after = getattr(e, "retry_after", None)
            if retry_after is not None:
                wait = float(retry_after)
            else:
                wait = min(base_delay * (2**attempt), 60.0)
            await asyncio.sleep(wait)
    raise last_error


class AnthropicProvider(BaseProvider):
    def __init__(self):
        self._client = AsyncAnthropic()

    def _resolve_model(self, model: str) -> str:
        if model in MODELS:
            return model
        mapping = {
            "claude-opus": "claude-opus-4-6",
            "claude-sonnet": "claude-sonnet-4-6",
            "claude-haiku": "claude-haiku-4-5",
        }
        return mapping.get(model, "claude-sonnet-4-6")

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[Tool],
        model: str,
        temperature: float,
        max_tokens: int,
        agent_name: str,
    ) -> tuple[Message, AgentEvent]:
        actual_model = self._resolve_model(model)
        system_prompt, anthropic_msgs = _convert_messages(messages)
        tool_schemas = _tools_to_anthropic_schema(tools) if tools else []

        async def _create():
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
            return await self._client.messages.create(**kwargs)

        start = time.time()
        response = await _with_retry(_create)
        latency = (time.time() - start) * 1000

        content_text, parsed_tool_calls = _parse_content_blocks(response.content)
        prices = PRICING.get(actual_model, {"input": 3.00, "output": 15.00})
        cost = (
            response.usage.input_tokens * prices["input"]
            + response.usage.output_tokens * prices["output"]
        ) / 1_000_000

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
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "has_tool_calls": parsed_tool_calls is not None,
            },
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            cost_usd=round(cost, 6),
            latency_ms=round(latency, 2),
        )
        return msg, event

    async def stream(
        self,
        messages: list[dict],
        tools: list[Tool],
        model: str,
        temperature: float,
        max_tokens: int,
        agent_name: str,
    ) -> AsyncGenerator[str | tuple[str, Message, AgentEvent], None]:
        actual_model = self._resolve_model(model)
        system_prompt, anthropic_msgs = _convert_messages(messages)
        tool_schemas = _tools_to_anthropic_schema(tools) if tools else []
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

        async def _do_stream():
            start = time.time()
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "type", None) == "text_delta":
                            text = getattr(delta, "text", None)
                            if text:
                                yield text
                    elif event.type == "text":
                        text = getattr(event, "text", None)
                        if text:
                            yield text
                final = await stream.get_final_message()
                latency = (time.time() - start) * 1000
                content_text, parsed_tool_calls = _parse_content_blocks(final.content)
                prices = PRICING.get(actual_model, {"input": 3.00, "output": 15.00})
                usage = getattr(final, "usage", None)
                input_tokens = usage.input_tokens if usage else 0
                output_tokens = usage.output_tokens if usage else 0
                cost = (
                    input_tokens * prices["input"] + output_tokens * prices["output"]
                ) / 1_000_000
                msg = Message(
                    role=Role.ASSISTANT,
                    content=content_text or None,
                    tool_calls=parsed_tool_calls,
                )
                evt = AgentEvent(
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
                if parsed_tool_calls:
                    yield ("tool_calls", msg, evt)
                else:
                    yield ("done", msg, evt)

        last_error = None
        for attempt in range(5):
            try:
                async for chunk in _do_stream():
                    yield chunk
                return
            except RateLimitError as e:
                last_error = e
                if attempt == 4:
                    raise
                retry_after = getattr(e, "retry_after", None)
                wait = (
                    float(retry_after)
                    if retry_after is not None
                    else min(1.0 * (2**attempt), 60.0)
                )
                await asyncio.sleep(wait)
        if last_error:
            raise last_error


def call_anthropic(
    messages: list[dict],
    tools: list[Tool],
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    provider = AnthropicProvider()
    return asyncio.run(
        provider.chat_completion(
            messages, tools, model, temperature, max_tokens, agent_name
        )
    )


def call_anthropic_stream(
    messages: list[dict],
    tools: list[Tool],
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
):
    async def _collect():
        provider = AnthropicProvider()
        chunks = []
        async for c in provider.stream(
            messages, tools, model, temperature, max_tokens, agent_name
        ):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(_collect())
    for c in chunks:
        yield c

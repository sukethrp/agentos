from __future__ import annotations
import json
import os
import time
from typing import AsyncGenerator
import httpx
from dotenv import load_dotenv
from agentos.core.types import Message, ToolCall, AgentEvent, Role
from agentos.core.tool import Tool
from agentos.providers.base import BaseProvider

load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _get_ollama_base_url() -> str:
    return os.getenv("AGENTOS_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def _tools_to_ollama_schema(tools: list[Tool]) -> list[dict]:
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
            }
        )
    return schemas


def _convert_messages(messages: list[dict]) -> list[dict]:
    ollama_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            ollama_messages.append({"role": "system", "content": content or ""})
        elif role == "tool":
            ollama_messages.append(
                {"role": "user", "content": f"Tool result: {content}"}
            )
        elif role == "assistant" and msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args.replace("'", '"')) if args else {}
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    {"function": {"name": fn.get("name", ""), "arguments": args}}
                )
            ollama_messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
            )
        else:
            ollama_messages.append({"role": role, "content": content or ""})
    return ollama_messages


def _parse_ollama_tool_calls(raw: list) -> list[ToolCall]:
    result = []
    for tc in raw:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        result.append(ToolCall(name=name, arguments=args))
    return result


class OllamaProvider(BaseProvider):
    def __init__(self, ollama_base_url: str | None = None):
        self._base_url = (ollama_base_url or _get_ollama_base_url()).rstrip("/")
        self._client = httpx.AsyncClient(timeout=120.0)

    async def _validate_model(self, model: str) -> None:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise ConnectionError(
                "Cannot connect to Ollama. Make sure Ollama is running: 'ollama serve'"
            ) from e
        data = resp.json()
        models = data.get("models", [])
        for m in models:
            name = m.get("name", "")
            if model == name or model == name.split(":")[0]:
                return
        raise ValueError(
            f"Model '{model}' not found. Available: {[m.get('name') for m in models]}"
        )

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[Tool],
        model: str,
        temperature: float,
        max_tokens: int,
        agent_name: str,
    ) -> tuple[Message, AgentEvent]:
        await self._validate_model(model)
        ollama_messages = _convert_messages(messages)
        tool_schemas = _tools_to_ollama_schema(tools) if tools else []
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
            start = time.time()
            resp = await self._client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - start) * 1000
        except httpx.ConnectError as e:
            raise ConnectionError(
                "Cannot connect to Ollama. Make sure Ollama is running: 'ollama serve'"
            ) from e
        message_data = data.get("message", {})
        content = message_data.get("content", "")
        raw_tool_calls = message_data.get("tool_calls", [])
        parsed_tool_calls = (
            _parse_ollama_tool_calls(raw_tool_calls) if raw_tool_calls else None
        )
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
            cost_usd=0.0,
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
        await self._validate_model(model)
        ollama_messages = _convert_messages(messages)
        tool_schemas = _tools_to_ollama_schema(tools) if tools else []
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tool_schemas:
            payload["tools"] = tool_schemas
        try:
            start = time.time()
            async with self._client.stream(
                "POST", f"{self._base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                content_parts = []
                tool_calls_acc: dict[int, dict] = {}
                chunk: dict = {}
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg_data = chunk.get("message", {})
                    if msg_data.get("content"):
                        content_parts.append(msg_data["content"])
                        yield msg_data["content"]
                    raw_tcs = msg_data.get("tool_calls", [])
                    for i, tc in enumerate(raw_tcs):
                        fn = tc.get("function", {})
                        if i not in tool_calls_acc:
                            tool_calls_acc[i] = {"name": "", "arguments": {}}
                        acc = tool_calls_acc[i]
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                        args = fn.get("arguments", {})
                        if isinstance(args, dict):
                            acc["arguments"].update(args)
                        elif isinstance(args, str):
                            try:
                                parsed = json.loads(args) if args else {}
                                acc["arguments"].update(parsed)
                            except json.JSONDecodeError:
                                pass
                latency = (time.time() - start) * 1000
                full_content = "".join(content_parts)
                parsed_tool_calls = None
                if tool_calls_acc:
                    parsed_tool_calls = []
                    for idx in sorted(tool_calls_acc.keys()):
                        acc = tool_calls_acc[idx]
                        if acc["name"]:
                            parsed_tool_calls.append(
                                ToolCall(name=acc["name"], arguments=acc["arguments"])
                            )
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)
                msg = Message(
                    role=Role.ASSISTANT,
                    content=full_content or None,
                    tool_calls=parsed_tool_calls,
                )
                evt = AgentEvent(
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
                    cost_usd=0.0,
                    latency_ms=round(latency, 2),
                )
                if parsed_tool_calls:
                    yield ("tool_calls", msg, evt)
                else:
                    yield ("done", msg, evt)
        except httpx.ConnectError as e:
            raise ConnectionError(
                "Cannot connect to Ollama. Make sure Ollama is running: 'ollama serve'"
            ) from e


def call_ollama(
    messages: list[dict],
    tools: list[Tool],
    model: str = "llama3.1",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    import asyncio

    provider = OllamaProvider()
    return asyncio.run(
        provider.chat_completion(
            messages, tools, model, temperature, max_tokens, agent_name
        )
    )


def call_ollama_stream(
    messages: list[dict],
    tools: list[Tool],
    model: str = "llama3.1",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
):
    import asyncio

    async def _collect():
        provider = OllamaProvider()
        chunks = []
        async for c in provider.stream(
            messages, tools, model, temperature, max_tokens, agent_name
        ):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(_collect())
    for c in chunks:
        yield c

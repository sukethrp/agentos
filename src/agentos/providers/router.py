"""Model Router — automatically picks the right provider based on model name.

Usage:
    from agentos.providers.router import call_model

    # These all work:
    call_model("gpt-4o-mini", messages, tools)        # → OpenAI
    call_model("claude-sonnet", messages, tools)       # → Anthropic
    call_model("ollama:llama3.1", messages, tools)     # → Ollama (local)
"""

from __future__ import annotations
from typing import Generator
from agentos.core.types import Message, AgentEvent
from agentos.core.tool import Tool


# Provider registry
OPENAI_MODELS = {
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1-mini",
    "o3-mini",
}
ANTHROPIC_MODELS = {
    "claude-sonnet",
    "claude-haiku",
    "claude-opus",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
}


def detect_provider(model: str) -> str:
    """Detect which provider to use based on model name."""
    if (
        model in OPENAI_MODELS
        or model.startswith("gpt-")
        or model.startswith("o1")
        or model.startswith("o3")
    ):
        return "openai"
    elif model in ANTHROPIC_MODELS or model.startswith("claude"):
        return "anthropic"
    elif model.startswith("ollama:"):
        return "ollama"
    else:
        return "openai"  # default fallback


def call_model(
    model: str,
    messages: list[dict],
    tools: list[Tool],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    """Route to the correct provider based on model name."""
    provider = detect_provider(model)

    if provider == "openai":
        from agentos.providers.openai_provider import call_llm

        return call_llm(
            messages,
            tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )

    elif provider == "anthropic":
        from agentos.providers.anthropic_provider import call_anthropic

        return call_anthropic(
            messages,
            tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )

    elif provider == "ollama":
        from agentos.providers.ollama_provider import call_ollama

        actual_model = model.replace("ollama:", "")
        return call_ollama(
            messages,
            tools,
            model=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )

    else:
        raise ValueError(f"Unknown provider for model: {model}")


def call_model_stream(
    model: str,
    messages: list[dict],
    tools: list[Tool],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> Generator[str | tuple[str, Message, AgentEvent], None, None]:
    provider = detect_provider(model)

    if provider == "openai":
        from agentos.providers.openai_provider import call_llm_stream

        yield from call_llm_stream(
            messages,
            tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )
    elif provider == "anthropic":
        from agentos.providers.anthropic_provider import call_anthropic_stream

        yield from call_anthropic_stream(
            messages,
            tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )
    elif provider == "ollama":
        from agentos.providers.ollama_provider import call_ollama_stream

        actual_model = model.replace("ollama:", "")
        yield from call_ollama_stream(
            messages,
            tools,
            model=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )
    else:
        msg, event = call_model(
            model,
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )
        if msg.tool_calls:
            yield ("tool_calls", msg, event)
        else:
            yield ("done", msg, event)


def list_providers() -> dict:
    """List all supported providers and models."""
    return {
        "openai": list(OPENAI_MODELS),
        "anthropic": list(ANTHROPIC_MODELS),
        "ollama": [
            "ollama:<any-model>",
            "ollama:llama3.1",
            "ollama:mistral",
            "ollama:gemma2",
        ],
    }

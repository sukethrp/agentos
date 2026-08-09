"""Model Router — automatically picks the right provider based on model name.

Usage:
    from agentos.providers.router import call_model

    # These all work:
    call_model("gpt-4o-mini", messages, tools)        # → OpenAI
    call_model("claude-sonnet", messages, tools)       # → Anthropic
    call_model("ollama:llama3.1", messages, tools)     # → Ollama (local)
"""

from __future__ import annotations

from collections.abc import Generator

from agentos.core.tool import Tool
from agentos.core.types import AgentEvent, Message
from agentos.logging import get_correlation, get_logger
from agentos.replay.provider import record_completion, record_stream
from agentos.replay.schema import SeamKind
from agentos.replay.seam import call_site_id

_log = get_logger("agentos.providers")

# Call site ids are (module, qualname, seam, label), never line numbers, so
# reformatting this file does not invalidate the trace corpus.
CS_CALL_MODEL = call_site_id(__name__, "call_model", SeamKind.PROVIDER)
CS_CALL_MODEL_STREAM = call_site_id(__name__, "call_model_stream", SeamKind.PROVIDER)


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


def resolve_provider(model: str) -> str:
    """The backend that will actually serve this call.

    Not the same as `detect_provider`: demo mode overrides the model name and
    routes everything to the mock. Recording the detected provider instead of
    the resolved one would describe a call that never happened.
    """
    from agentos.demo import is_demo_mode

    return "mock" if is_demo_mode() else detect_provider(model)


def call_model(
    model: str,
    messages: list[dict],
    tools: list[Tool],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    """Route to the correct provider based on model name.

    The single choke point for non-streaming completions, and therefore the
    PROVIDER seam for all backends at once.
    """
    provider = resolve_provider(model)
    return record_completion(
        CS_CALL_MODEL,
        provider=provider,
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
        thunk=lambda: _dispatch_completion(
            provider, model, messages, tools, temperature, max_tokens, agent_name
        ),
    )


def _dispatch_completion(
    provider: str,
    model: str,
    messages: list[dict],
    tools: list[Tool],
    temperature: float,
    max_tokens: int,
    agent_name: str,
) -> tuple[Message, AgentEvent]:
    if provider == "mock":
        from agentos.providers.mock import call_mock

        return call_mock(
            messages, tools, model=model, temperature=temperature,
            max_tokens=max_tokens, agent_name=agent_name,
        )

    _log.debug(
        "provider.route",
        extra={**get_correlation(), "provider": provider, "model": model},
    )

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
    provider = resolve_provider(model)
    yield from record_stream(
        CS_CALL_MODEL_STREAM,
        provider=provider,
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
        thunk=lambda: _dispatch_stream(
            provider, model, messages, tools, temperature, max_tokens, agent_name
        ),
    )


def _dispatch_stream(
    provider: str,
    model: str,
    messages: list[dict],
    tools: list[Tool],
    temperature: float,
    max_tokens: int,
    agent_name: str,
) -> Generator[str | tuple[str, Message, AgentEvent], None, None]:
    if provider == "mock":
        from agentos.providers.mock import call_mock_stream

        yield from call_mock_stream(
            messages, tools, model=model, temperature=temperature,
            max_tokens=max_tokens, agent_name=agent_name,
        )
        return

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
        # Deliberately _dispatch_completion and not call_model: this is already
        # inside the stream seam, and going back through call_model would record
        # a second nested PROVIDER event for one logical call.
        msg, event = _dispatch_completion(
            provider, model, messages, tools, temperature, max_tokens, agent_name
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

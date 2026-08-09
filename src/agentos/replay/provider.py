"""Codec and interception helpers for the PROVIDER seam.

Layering. This is the one module under `agentos.replay` that imports
`agentos.core`, and it is deliberately NOT re-exported from
`agentos.replay.__init__`. Importing `agentos.replay` stays stdlib-only per
ADR-008; you pay for pydantic only by importing `agentos.replay.provider`
explicitly, which only the provider layer does.

Why a codec exists at all. The blob store is JSON by choice: a trace recorded
today must still load after pydantic changes how it serializes, so traces never
hold pickled objects and `canonical_json` stays strict about what it will
encode. But this seam's payloads are pydantic models (`Message`, `AgentEvent`)
and `Tool` objects, none of which are JSON. So the projection lives here, in
both directions, and `canonical_json` is left alone.

What gets digested. Exactly the fields the router actually has:

    agent_name, max_tokens, messages, model, provider, temperature, tools

`top_p`, `seed`, `response_format`, and `stop` are NOT here, because no provider
in this repository accepts them. Digesting them as constants would claim a
coverage that does not exist, which is worse than the gap itself. When they are
plumbed through the provider signatures, add them to `PROVIDER_DIGEST_FIELDS`
and bump `PROVIDER_CODEC_VERSION`. Every existing trace then refuses to replay
against the new build instead of silently comparing digests taken over
different field sets. That refusal is the entire point of the fingerprint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from agentos.core.tool import Tool
from agentos.core.types import AgentEvent, Message

from .schema import SeamKind, digest_obj
from .seam import NullInterceptor, current_interceptor, intercept

__all__ = [
    "PROVIDER_CODEC_VERSION",
    "PROVIDER_DIGEST_FIELDS",
    "decode_completion",
    "decode_stream",
    "encode_completion",
    "encode_stream",
    "provider_codec_fingerprint",
    "provider_input",
    "provider_seam_codecs",
    "record_completion",
    "record_stream",
    "tracing_disabled",
]

StreamChunk = str | tuple[str, Message, AgentEvent]
Completion = tuple[Message, AgentEvent]

PROVIDER_CODEC_VERSION = "1"

# Sorted, and must stay in lockstep with the keys `provider_input` returns.
# `test_provider_seam.py` asserts that, so the fingerprint cannot drift away
# from what is actually hashed.
PROVIDER_DIGEST_FIELDS: tuple[str, ...] = (
    "agent_name",
    "max_tokens",
    "messages",
    "model",
    "provider",
    "temperature",
    "tools",
)


def provider_codec_fingerprint() -> str:
    """Identity of this seam's input projection plus codec version."""
    return digest_obj(
        {
            "codec_version": PROVIDER_CODEC_VERSION,
            "fields": sorted(PROVIDER_DIGEST_FIELDS),
        }
    )[:24]


def provider_seam_codecs() -> dict[str, str]:
    """The `seam_codecs` entry to put in a `RunHeader` for a recorded run."""
    return {SeamKind.PROVIDER.value: provider_codec_fingerprint()}


def tracing_disabled() -> bool:
    """True when no interceptor is installed, so the seam must get out of the way.

    Load bearing for streaming: without this the recording path would have to
    materialize the whole generator before yielding anything, which would turn
    token streaming into a single blocking call for every caller, traced or not.
    """
    return isinstance(current_interceptor(), NullInterceptor)


# ── Input projection ─────────────────────────────────────────────────────────


def provider_input(
    *,
    provider: str,
    model: str,
    messages: list[dict],
    tools: Iterable[Tool],
    temperature: float,
    max_tokens: int,
    agent_name: str,
) -> dict[str, Any]:
    """The object whose digest must fully determine the completion.

    `provider` is the backend that actually served the call, not the one the
    model name suggests. In demo mode those differ, and recording the suggested
    one would describe a call that never happened.

    Tools go through `ToolSpec.to_openai_schema()`, the same projection
    `call_llm` puts on the wire. Anthropic and Ollama reshape it before sending,
    but the semantic content that can change a completion is the tool's name,
    description, and parameters, which is exactly what this carries. A tool
    whose description is edited therefore produces a different digest, which is
    correct: the model sees different text.
    """
    return {
        "agent_name": agent_name,
        "max_tokens": max_tokens,
        "messages": messages,
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "tools": [t.spec.to_openai_schema() for t in tools] if tools else [],
    }


# ── Output codec ─────────────────────────────────────────────────────────────


def encode_completion(result: Completion) -> dict[str, Any]:
    msg, event = result
    return {
        "event": event.model_dump(mode="json"),
        "message": msg.model_dump(mode="json"),
    }


def decode_completion(payload: dict[str, Any]) -> Completion:
    """Rebuild real models, not dicts.

    `core.agent` reads `msg.tool_calls` and `msg.content`, so handing back the
    raw JSON would make replay diverge from recording at the first attribute
    access rather than at a seam, which is the worst place to find out.
    """
    return (
        Message.model_validate(payload["message"]),
        AgentEvent.model_validate(payload["event"]),
    )


def encode_stream(chunks: Iterable[StreamChunk]) -> dict[str, Any]:
    """Record chunk boundaries exactly as they arrived.

    Two surfaces, on purpose. `chunks` is what replay re-yields, so a WebSocket
    consumer sees the same boundaries it saw live. `text` is the concatenation,
    and it is what a human comparison should look at, because a provider that
    splits the same answer differently changed its chunking, not its answer.
    Boundaries are never normalized: normalizing would make the two runs compare
    equal and destroy the only evidence that the chunking changed.
    """
    encoded: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, str):
            encoded.append({"kind": "text", "text": chunk})
            text_parts.append(chunk)
            continue
        tag, msg, event = chunk
        encoded.append(
            {
                "kind": "final",
                "tag": tag,
                "message": msg.model_dump(mode="json"),
                "event": event.model_dump(mode="json"),
            }
        )
    return {"chunks": encoded, "text": "".join(text_parts)}


def decode_stream(payload: dict[str, Any]) -> list[StreamChunk]:
    out: list[StreamChunk] = []
    for chunk in payload["chunks"]:
        if chunk["kind"] == "text":
            out.append(chunk["text"])
            continue
        out.append(
            (
                chunk["tag"],
                Message.model_validate(chunk["message"]),
                AgentEvent.model_validate(chunk["event"]),
            )
        )
    return out


# ── Seam entry points ────────────────────────────────────────────────────────


def record_completion(
    call_site: str,
    *,
    provider: str,
    model: str,
    messages: list[dict],
    tools: Iterable[Tool],
    temperature: float,
    max_tokens: int,
    agent_name: str,
    thunk: Callable[[], Completion],
) -> Completion:
    """Route one completion through the PROVIDER seam."""
    if tracing_disabled():
        return thunk()
    payload = intercept(
        SeamKind.PROVIDER,
        call_site,
        provider_input(
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        ),
        lambda: encode_completion(thunk()),
        name=f"{provider}:{model}",
    )
    return decode_completion(payload)


def record_stream(
    call_site: str,
    *,
    provider: str,
    model: str,
    messages: list[dict],
    tools: Iterable[Tool],
    temperature: float,
    max_tokens: int,
    agent_name: str,
    thunk: Callable[[], Iterable[StreamChunk]],
) -> Iterator[StreamChunk]:
    """Route one streamed completion through the PROVIDER seam.

    Recording necessarily materializes the stream, because a chunk list cannot
    be written to a blob until it is complete. Untraced callers keep their lazy
    generator, which is why `tracing_disabled` is checked first.
    """
    if tracing_disabled():
        yield from thunk()
        return
    payload = intercept(
        SeamKind.PROVIDER,
        call_site,
        provider_input(
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_name=agent_name,
        ),
        lambda: encode_stream(list(thunk())),
        name=f"{provider}:{model}:stream",
    )
    yield from decode_stream(payload)

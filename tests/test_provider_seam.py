"""M1 gate for the PROVIDER seam.

`test_replay_roundtrip.py` proves the seam machinery works against a synthetic
provider. This file proves it works against the real router, which is where the
interesting failures live: pydantic payloads that the JSON blob store cannot
hold, streaming chunk boundaries, and a digest whose field set can silently
drift away from what the provider actually accepts.
"""

from __future__ import annotations

import pytest

from agentos.core.tool import Tool
from agentos.core.types import AgentEvent, Message
from agentos.providers import mock as mock_mod
from agentos.providers.router import call_model, call_model_stream
from agentos.replay import (
    BlobStore,
    DivergencePolicy,
    Recorder,
    Replayer,
    RunHeader,
    SeamCodecMismatch,
    SeamKind,
    TraceReader,
    TraceWriter,
    trace_digest,
    use_interceptor,
)
from agentos.replay.provider import (
    PROVIDER_DIGEST_FIELDS,
    provider_codec_fingerprint,
    provider_input,
    provider_seam_codecs,
    tracing_disabled,
)

MODEL = "gpt-4o-mini"
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]


def get_weather(location: str) -> str:
    """Look up the weather for a location."""
    return f"sunny in {location}"


@pytest.fixture
def tools() -> list[Tool]:
    return [Tool(get_weather)]


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    """Route every call through mock.call_mock, which is what demo mode does."""
    monkeypatch.setenv("AGENTOS_DEMO_MODE", "true")


@pytest.fixture
def live_calls(monkeypatch):
    """Counts real provider invocations so 'zero live calls' is measured, not assumed."""
    counter = {"completion": 0, "stream": 0}
    real_call, real_stream = mock_mod.call_mock, mock_mod.call_mock_stream

    def counting_call(*args, **kwargs):
        counter["completion"] += 1
        return real_call(*args, **kwargs)

    def counting_stream(*args, **kwargs):
        counter["stream"] += 1
        yield from real_stream(*args, **kwargs)

    monkeypatch.setattr(mock_mod, "call_mock", counting_call)
    monkeypatch.setattr(mock_mod, "call_mock_stream", counting_stream)
    return counter


def _record(tmp_path, fn):
    header = RunHeader.new(seam_codecs=provider_seam_codecs(), labels={"suite": "m1"})
    with TraceWriter(tmp_path, header) as w:
        rec = Recorder(w)
        with use_interceptor(rec):
            result = fn()
    return header, rec, result


def _replayer(tmp_path, header, policy=DivergencePolicy.STRICT, codecs=None):
    reader = TraceReader(tmp_path / "runs" / f"{header.run_id}.jsonl")
    return Replayer(
        reader.events,
        BlobStore(tmp_path),
        policy=policy,
        header=reader.header,
        codecs=provider_seam_codecs() if codecs is None else codecs,
    )


# ── The contract ─────────────────────────────────────────────────────────────


def test_record_then_replay_through_call_model(tmp_path, tools, live_calls):
    """The M1 gate: identical trace digest, zero live calls, real models back."""
    header, rec, live_result = _record(
        tmp_path,
        lambda: call_model(MODEL, MESSAGES, tools, agent_name="seam-test"),
    )
    assert live_calls["completion"] == 1
    assert [e.seam for e in rec.events] == [SeamKind.PROVIDER]

    rp = _replayer(tmp_path, header)
    with use_interceptor(rp):
        replayed_result = call_model(MODEL, MESSAGES, tools, agent_name="seam-test")

    assert live_calls["completion"] == 1, "replay must not touch the provider"
    assert rp.live_calls == 0
    assert not rp.tainted
    assert trace_digest(rp.consumed) == trace_digest(rec.events)

    live_msg, live_event = live_result
    replayed_msg, replayed_event = replayed_result
    # Decoded back into real models, because core.agent reads attributes off
    # these rather than treating them as dicts.
    assert isinstance(replayed_msg, Message)
    assert isinstance(replayed_event, AgentEvent)
    assert replayed_msg.model_dump() == live_msg.model_dump()
    assert replayed_event.model_dump() == live_event.model_dump()


def test_two_live_runs_actually_differ(tmp_path, tools):
    """Guards the guard: if the mock were deterministic the gate proves nothing."""
    _h1, rec1, _ = _record(
        tmp_path / "a", lambda: call_model(MODEL, MESSAGES, tools, agent_name="x")
    )
    _h2, rec2, _ = _record(
        tmp_path / "b", lambda: call_model(MODEL, MESSAGES, tools, agent_name="x")
    )
    out_a = BlobStore(tmp_path / "a").get_obj(rec1.events[0].output_ref)
    out_b = BlobStore(tmp_path / "b").get_obj(rec2.events[0].output_ref)
    assert out_a != out_b
    # Same inputs though, so the digests must agree or replay could never match.
    assert rec1.events[0].input_digest == rec2.events[0].input_digest


def test_changed_prompt_is_a_divergence_not_a_live_call(tmp_path, tools, live_calls):
    from agentos.replay import DivergenceError

    header, _rec, _ = _record(
        tmp_path, lambda: call_model(MODEL, MESSAGES, tools, agent_name="x")
    )
    poisoned = [dict(m) for m in MESSAGES]
    poisoned[-1]["content"] = "What is the weather in Paris?"

    rp = _replayer(tmp_path, header)
    with use_interceptor(rp), pytest.raises(DivergenceError) as ei:
        call_model(MODEL, poisoned, tools, agent_name="x")
    assert "input mismatch" in str(ei.value)
    assert live_calls["completion"] == 1, "STRICT must not fall through to live"


# ── Streaming ────────────────────────────────────────────────────────────────


def test_stream_chunk_boundaries_replay_identically(tmp_path, live_calls):
    """Boundaries are replayed as recorded, not renormalized into one blob."""
    header, rec, live_chunks = _record(
        tmp_path,
        lambda: list(call_model_stream(MODEL, MESSAGES, [], agent_name="streamer")),
    )
    text_chunks = [c for c in live_chunks if isinstance(c, str)]
    assert len(text_chunks) > 1, "need real boundaries for this test to mean anything"

    rp = _replayer(tmp_path, header)
    with use_interceptor(rp):
        replayed_chunks = list(
            call_model_stream(MODEL, MESSAGES, [], agent_name="streamer")
        )

    assert live_calls["stream"] == 1
    assert rp.live_calls == 0
    assert trace_digest(rp.consumed) == trace_digest(rec.events)

    # Boundary-for-boundary, not just the concatenation.
    live_text = [c for c in live_chunks if isinstance(c, str)]
    replayed_text = [c for c in replayed_chunks if isinstance(c, str)]
    assert replayed_text == live_text
    assert len(replayed_chunks) == len(live_chunks)

    live_final = [c for c in live_chunks if not isinstance(c, str)]
    replayed_final = [c for c in replayed_chunks if not isinstance(c, str)]
    assert [t for t, _m, _e in replayed_final] == [t for t, _m, _e in live_final]
    for (_t, live_msg, _le), (_t2, rep_msg, _re) in zip(
        live_final, replayed_final, strict=True
    ):
        assert isinstance(rep_msg, Message)
        assert rep_msg.model_dump() == live_msg.model_dump()


def test_stream_blob_keeps_chunks_and_concatenated_text(tmp_path):
    """`chunks` is the replay surface, `text` is the comparison surface."""
    _header, rec, live_chunks = _record(
        tmp_path,
        lambda: list(call_model_stream(MODEL, MESSAGES, [], agent_name="streamer")),
    )
    payload = BlobStore(tmp_path).get_obj(rec.events[0].output_ref)
    assert payload["text"] == "".join(c for c in live_chunks if isinstance(c, str))
    assert [c["text"] for c in payload["chunks"] if c["kind"] == "text"] == [
        c for c in live_chunks if isinstance(c, str)
    ]


# ── The loud gap: seam codec fingerprints ────────────────────────────────────


def test_digest_fields_match_what_is_actually_hashed(tools):
    """Stops the fingerprint from drifting away from the real projection."""
    keys = provider_input(
        provider="mock",
        model=MODEL,
        messages=MESSAGES,
        tools=tools,
        temperature=0.7,
        max_tokens=1024,
        agent_name="x",
    ).keys()
    assert sorted(keys) == sorted(PROVIDER_DIGEST_FIELDS)


def test_absent_sampling_params_are_not_silently_digested(tools):
    """The known gap, asserted so it cannot be forgotten or quietly widened."""
    projection = provider_input(
        provider="mock",
        model=MODEL,
        messages=MESSAGES,
        tools=tools,
        temperature=0.7,
        max_tokens=1024,
        agent_name="x",
    )
    for absent in ("top_p", "seed", "response_format", "stop"):
        assert absent not in projection


def test_codec_fingerprint_mismatch_refuses_to_compare(tmp_path, tools):
    header, _rec, _ = _record(
        tmp_path, lambda: call_model(MODEL, MESSAGES, tools, agent_name="x")
    )
    stale = {SeamKind.PROVIDER.value: "fingerprint-from-an-older-build"}
    with pytest.raises(SeamCodecMismatch) as ei:
        _replayer(tmp_path, header, codecs=stale)

    msg = str(ei.value)
    assert "provider" in msg
    assert "Re-record the trace" in msg
    assert ei.value.recorded == provider_codec_fingerprint()
    assert ei.value.current == "fingerprint-from-an-older-build"


def test_missing_codec_for_a_recorded_seam_refuses(tmp_path, tools):
    header, _rec, _ = _record(
        tmp_path, lambda: call_model(MODEL, MESSAGES, tools, agent_name="x")
    )
    with pytest.raises(SeamCodecMismatch) as ei:
        _replayer(tmp_path, header, codecs={})
    assert "provides no codec" in str(ei.value)


def test_legacy_trace_without_codecs_still_loads(tmp_path, tools):
    """0.1.0 traces default to no codecs. Unknown must not mean mismatched."""
    header = RunHeader.new()  # no seam_codecs, as a pre-0.2.0 trace reads back
    assert header.seam_codecs == {}
    with TraceWriter(tmp_path, header) as w:
        rec = Recorder(w)
        with use_interceptor(rec):
            call_model(MODEL, MESSAGES, tools, agent_name="x")

    rp = _replayer(tmp_path, header)  # must not raise
    with use_interceptor(rp):
        call_model(MODEL, MESSAGES, tools, agent_name="x")
    assert trace_digest(rp.consumed) == trace_digest(rec.events)


# ── Replay provenance ────────────────────────────────────────────────────────


def test_replayed_runs_are_marked_in_the_header():
    """A replayed run carries the original latencies, so it must be flagged."""
    original = RunHeader.new(seam_codecs=provider_seam_codecs())
    assert not original.is_replay
    assert original.replayed_from is None

    derived = original.derive_replay()
    assert derived.is_replay
    assert derived.replayed_from == original.run_id
    assert derived.run_id != original.run_id
    assert derived.seam_codecs == original.seam_codecs


# ── No interceptor installed ─────────────────────────────────────────────────


def test_untraced_calls_are_transparent(tools, live_calls):
    assert tracing_disabled()
    msg, event = call_model(MODEL, MESSAGES, tools, agent_name="x")
    assert isinstance(msg, Message)
    assert isinstance(event, AgentEvent)
    assert live_calls["completion"] == 1


def test_untraced_streaming_stays_lazy(live_calls):
    """Recording materializes the stream; untraced callers must not pay that."""
    gen = call_model_stream(MODEL, MESSAGES, [], agent_name="x")
    assert live_calls["stream"] == 0, "generator must not run before iteration"
    first = next(gen)
    assert isinstance(first, str)
    assert live_calls["stream"] == 1
    gen.close()

"""M0 acceptance gate.

If these pass, the determinism contract holds for a single-threaded run with one
seam. Everything after this (clock freezing, async scheduling, diff, bisect) is
built on the guarantee proven here: record then replay yields an identical
trace digest, with zero live calls.
"""

from __future__ import annotations

import random

import pytest

from agentos.replay import (
    BlobStore,
    DivergenceError,
    DivergencePolicy,
    EventStatus,
    Recorder,
    ReplayedError,
    Replayer,
    RunHeader,
    SeamKind,
    TraceReader,
    TraceWriter,
    call_site_id,
    intercept,
    trace_digest,
    use_interceptor,
)

CS_COMPLETE = call_site_id(__name__, "FakeProvider.complete", SeamKind.PROVIDER)
CS_TOOL = call_site_id(__name__, "search_tool", SeamKind.TOOL)


class FakeProvider:
    """Nondeterministic on purpose. Two live runs must never agree."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> dict:
        self.calls += 1
        return {
            "text": f"answer to {prompt}",
            "nonce": random.random(),  # entropy
            "usage": {"in": len(prompt), "out": 7},
        }


def run_agent(
    provider: FakeProvider, steps: int = 3, poison: bool = False
) -> list[str]:
    """A tiny agent loop: provider call, tool call, provider call, ..."""
    out: list[str] = []
    for i in range(steps):
        prompt = f"step-{i}" + ("!" if poison and i == 1 else "")
        resp = intercept(
            SeamKind.PROVIDER,
            CS_COMPLETE,
            {"prompt": prompt},
            lambda p=prompt: provider.complete(p),
            name="complete",
        )
        out.append(resp["text"])
        hits = intercept(
            SeamKind.TOOL,
            CS_TOOL,
            {"q": resp["text"]},
            lambda t=resp["text"]: {"hits": [t.upper()]},
            name="search",
        )
        out.append(hits["hits"][0])
    return out


def record(tmp_path, steps: int = 3, poison: bool = False):
    provider = FakeProvider()
    header = RunHeader.new(labels={"suite": "m0"})
    with TraceWriter(tmp_path, header) as w:
        rec = Recorder(w)
        with use_interceptor(rec):
            result = run_agent(provider, steps=steps, poison=poison)
    return header, rec, result, provider


def replay(tmp_path, header, policy=DivergencePolicy.STRICT, poison=False, steps=3):
    reader = TraceReader(tmp_path / "runs" / f"{header.run_id}.jsonl")
    rp = Replayer(reader.events, BlobStore(tmp_path), policy=policy)
    provider = FakeProvider()
    with use_interceptor(rp):
        result = run_agent(provider, steps=steps, poison=poison)
    return rp, result, provider


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def test_record_then_replay_is_bit_identical(tmp_path):
    header, rec, live_result, _live_provider = record(tmp_path)
    rp, replayed_result, replay_provider = replay(tmp_path, header)

    assert replayed_result == live_result
    assert replay_provider.calls == 0, "replay must not touch the provider"
    assert len(rp.consumed) == len(rec.events)
    assert trace_digest(rp.consumed) == trace_digest(rec.events)
    assert not rp.tainted


def test_two_live_runs_actually_differ(tmp_path):
    """Guards the guard. If the fake were deterministic the test above is vacuous."""
    _h1, r1, _, _ = record(tmp_path / "a")
    _h2, r2, _, _ = record(tmp_path / "b")
    blobs_a = BlobStore(tmp_path / "a")
    blobs_b = BlobStore(tmp_path / "b")
    provider_a = [e for e in r1.events if e.seam is SeamKind.PROVIDER]
    provider_b = [e for e in r2.events if e.seam is SeamKind.PROVIDER]
    out_a = [blobs_a.get_obj(e.output_ref) for e in provider_a]
    out_b = [blobs_b.get_obj(e.output_ref) for e in provider_b]
    assert out_a != out_b


def test_wall_clock_is_not_part_of_equivalence(tmp_path):
    _header, rec, _, _ = record(tmp_path)
    before = trace_digest(rec.events)
    for ev in rec.events:
        ev.wall_start_ns += 10_000_000
        ev.wall_end_ns += 99_000_000
    assert trace_digest(rec.events) == before


def test_input_mismatch_raises_and_points_upstream(tmp_path):
    header, _, _, _ = record(tmp_path)
    with pytest.raises(DivergenceError) as ei:
        replay(tmp_path, header, poison=True)
    assert "input mismatch" in str(ei.value)
    assert ei.value.expected != ei.value.actual


def test_lenient_policy_taints_instead_of_raising(tmp_path):
    header, _, _, _ = record(tmp_path)
    rp, _, provider = replay(
        tmp_path, header, policy=DivergencePolicy.LENIENT, poison=True
    )
    assert rp.tainted
    assert rp.live_calls >= 1
    assert provider.calls >= 1


def test_unrecorded_call_site_diverges(tmp_path):
    """Control flow ran longer than the recording. Classic replay miss."""
    header, _, _, _ = record(tmp_path, steps=2)
    with pytest.raises(DivergenceError) as ei:
        replay(tmp_path, header, steps=4)
    assert "unrecorded call" in str(ei.value)


def test_errors_are_recorded_and_replayed(tmp_path):
    cs = call_site_id(__name__, "boom", SeamKind.PROVIDER)

    def boom():
        raise ValueError("provider 503")

    header = RunHeader.new()
    with TraceWriter(tmp_path, header) as w:
        rec = Recorder(w)
        with use_interceptor(rec), pytest.raises(ValueError):
            intercept(SeamKind.PROVIDER, cs, {"p": 1}, boom)

    assert rec.events[0].status is EventStatus.ERROR
    reader = TraceReader(tmp_path / "runs" / f"{header.run_id}.jsonl")
    rp = Replayer(reader.events, BlobStore(tmp_path))
    with use_interceptor(rp), pytest.raises(ReplayedError) as ei:
        intercept(SeamKind.PROVIDER, cs, {"p": 1}, boom)
    assert ei.value.error_type == "ValueError"


def test_blobs_dedupe(tmp_path):
    store = BlobStore(tmp_path)
    a = store.put_obj({"prompt": "you are a helpful assistant"})
    b = store.put_obj({"prompt": "you are a helpful assistant"})
    assert a == b
    assert store.get_obj(a)["prompt"].startswith("you are")


def test_header_rejects_future_major_version(tmp_path):
    from agentos.replay.schema import RunHeader as RH

    bad = RunHeader.new().to_dict() | {"schema_version": "9.0.0"}
    with pytest.raises(ValueError):
        RH.from_dict(bad)

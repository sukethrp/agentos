"""Structural alignment of two traces.

Invariant this file protects: one extra call at a site is one insertion, not
N `input_changed` reports. Aligning on `(seam, call_site, ordinal)` — the
replay key — is what produces the N-report bug, because every later ordinal
at that site shifts. Identity-key alignment does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentos.replay import (
    SCHEMA_VERSION,
    BlobStore,
    EventStatus,
    Recorder,
    RunHeader,
    SeamKind,
    TraceEvent,
    TraceReader,
    TraceWriter,
    call_site_id,
    intercept,
    use_interceptor,
)
from agentos.replay.diff import (
    ChangeKind,
    DiffReport,
    IncomparableError,
    compare_paths,
    compare_readers,
    diff_events,
    identity_key,
    render_human,
)
from agentos.replay.schema import digest_obj


CS_P = "site-provider"
CS_T = "site-tool"
_DEFAULT_OUTPUT = "b2b:" + "ab" * 32


def make_event(
    seq: int,
    *,
    seam: SeamKind = SeamKind.PROVIDER,
    call_site: str = CS_P,
    ordinal: int | None = None,
    agent_id: str = "root",
    lamport: int | None = None,
    input_digest: str | None = None,
    output_ref: str | None | object = ...,
    status: EventStatus = EventStatus.OK,
    error_type: str | None = None,
) -> TraceEvent:
    digest = input_digest if input_digest is not None else f"b2b:{seq:064x}"
    if output_ref is ...:
        output_ref = None if status is EventStatus.ERROR else _DEFAULT_OUTPUT
    return TraceEvent(
        event_id=f"e{seq}-{agent_id}",
        run_id="run",
        seq=seq,
        seam=seam,
        call_site=call_site,
        ordinal=seq if ordinal is None else ordinal,
        input_digest=digest,
        agent_id=agent_id,
        lamport=seq if lamport is None else lamport,
        output_ref=output_ref,  # type: ignore[arg-type]
        status=status,
        error_type=error_type,
    )


def _ordinal_align_input_changes(
    left: list[TraceEvent], right: list[TraceEvent]
) -> int:
    """The algorithm that is wrong, kept here so the test can name it."""
    index = {(e.seam, e.call_site, e.ordinal): e for e in right}
    n = 0
    for event in left:
        matched = index.get((event.seam, event.call_site, event.ordinal))
        if matched is not None and matched.input_digest != event.input_digest:
            n += 1
    return n


def _write_trace(
    root: Path,
    events: list[TraceEvent],
    *,
    header_kwargs: dict | None = None,
    inputs: dict[str, object] | None = None,
) -> Path:
    kwargs = dict(header_kwargs or {})
    if inputs:
        kwargs.setdefault("stores_input_blobs", True)
    header = RunHeader.new(**kwargs)
    with TraceWriter(root, header) as writer:
        for event in events:
            event.run_id = header.run_id
            writer.append(event)
    if inputs:
        store = BlobStore(root)
        for obj in inputs.values():
            store.put_obj(obj)
    return root / "runs" / f"{header.run_id}.jsonl"


# ── Self-diff is empty ───────────────────────────────────────────────────────


def test_self_diff_is_empty_for_several_shapes():
    mixed = [
        make_event(1, call_site=CS_P, ordinal=0, input_digest="b2b:" + "a" * 64),
        make_event(
            2,
            seam=SeamKind.TOOL,
            call_site=CS_T,
            ordinal=0,
            input_digest="b2b:" + "t" * 64,
        ),
        make_event(3, call_site=CS_P, ordinal=1, input_digest="b2b:" + "b" * 64),
    ]
    error = [
        make_event(
            0,
            status=EventStatus.ERROR,
            error_type="ValueError",
            output_ref=None,
            input_digest="b2b:" + "e" * 64,
        )
    ]
    streaming = [
        make_event(
            0,
            call_site="site-stream",
            input_digest="b2b:" + "s" * 64,
            output_ref="b2b:" + "c" * 64,
        )
    ]
    multi_agent = [
        make_event(1, agent_id="root", ordinal=0),
        make_event(
            2, agent_id="worker", ordinal=0, call_site=CS_T, seam=SeamKind.TOOL
        ),
        make_event(3, agent_id="root", ordinal=1),
    ]
    for shape in ([], mixed, error, streaming, multi_agent):
        report = diff_events(shape, list(shape))
        assert report.identical
        assert report.changes == ()
        assert report.first_divergence is None
        assert report.n_unchanged == len(shape)
        assert report.n_input_changed == 0
        assert report.n_output_changed == 0
        assert report.n_status_changed == 0
        assert report.n_inserted == 0
        assert report.n_deleted == 0


def test_self_diff_of_a_recorded_error_trace(tmp_path):
    cs = call_site_id(__name__, "boom", SeamKind.PROVIDER)

    def boom():
        raise ValueError("provider 503")

    header = RunHeader.new()
    with TraceWriter(tmp_path, header) as writer:
        rec = Recorder(writer)
        with use_interceptor(rec), pytest.raises(ValueError):
            intercept(SeamKind.PROVIDER, cs, {"p": 1}, boom)
    report = diff_events(rec.events, rec.events)
    assert report.identical
    assert report.changes == ()


# ── The test that matters most ───────────────────────────────────────────────


def test_insert_at_head_of_call_site_is_one_insertion_not_n_input_changed():
    """One extra first call at a site must not shift later hits into changes."""
    d_new, d_a, d_b, d_c = (f"b2b:{n:064x}" for n in (11, 21, 22, 23))
    d_t0, d_t1 = (f"b2b:{n:064x}" for n in (31, 32))

    left = [
        make_event(1, call_site=CS_P, ordinal=0, input_digest=d_a),
        make_event(
            2, seam=SeamKind.TOOL, call_site=CS_T, ordinal=0, input_digest=d_t0
        ),
        make_event(3, call_site=CS_P, ordinal=1, input_digest=d_b),
        make_event(
            4, seam=SeamKind.TOOL, call_site=CS_T, ordinal=1, input_digest=d_t1
        ),
        make_event(5, call_site=CS_P, ordinal=2, input_digest=d_c),
    ]
    right = [
        make_event(1, call_site=CS_P, ordinal=0, input_digest=d_new),
        make_event(2, call_site=CS_P, ordinal=1, input_digest=d_a),
        make_event(
            3, seam=SeamKind.TOOL, call_site=CS_T, ordinal=0, input_digest=d_t0
        ),
        make_event(4, call_site=CS_P, ordinal=2, input_digest=d_b),
        make_event(
            5, seam=SeamKind.TOOL, call_site=CS_T, ordinal=1, input_digest=d_t1
        ),
        make_event(6, call_site=CS_P, ordinal=3, input_digest=d_c),
    ]

    assert _ordinal_align_input_changes(left, right) == 3, (
        "guard: the ordinal aligner must still be wrong on this fixture, "
        "otherwise the test no longer proves the identity-key choice"
    )

    report = diff_events(left, right)
    assert not report.identical
    assert report.n_inserted == 1
    assert report.n_input_changed == 0
    assert report.n_deleted == 0
    assert report.n_output_changed == 0
    assert report.n_status_changed == 0
    assert len(report.changes) == 1
    change = report.changes[0]
    assert change.kind is ChangeKind.INSERTED
    assert change.right is not None
    assert change.right["input_digest"] == d_new
    assert report.first_divergence is not None
    assert report.first_divergence.kind is ChangeKind.INSERTED


# ── One test per classification kind ─────────────────────────────────────────


def test_input_changed_on_aligned_pair():
    left = [make_event(1, ordinal=0, input_digest="b2b:" + "a" * 64)]
    right = [make_event(1, ordinal=0, input_digest="b2b:" + "b" * 64)]
    report = diff_events(left, right)
    assert report.n_input_changed == 1
    assert report.changes[0].kind is ChangeKind.INPUT_CHANGED
    assert report.first_divergence is not None
    assert "input_changed" in report.first_divergence.message


def test_output_changed_on_aligned_pair():
    digest = "b2b:" + "a" * 64
    left = [
        make_event(1, ordinal=0, input_digest=digest, output_ref="b2b:" + "1" * 64)
    ]
    right = [
        make_event(1, ordinal=0, input_digest=digest, output_ref="b2b:" + "2" * 64)
    ]
    report = diff_events(left, right)
    assert report.n_output_changed == 1
    assert report.n_input_changed == 0
    assert report.changes[0].kind is ChangeKind.OUTPUT_CHANGED


def test_status_changed_on_aligned_pair():
    digest = "b2b:" + "a" * 64
    out = "b2b:" + "1" * 64
    left = [
        make_event(
            1, ordinal=0, input_digest=digest, output_ref=out, status=EventStatus.OK
        )
    ]
    right = [
        make_event(
            1,
            ordinal=0,
            input_digest=digest,
            output_ref=out,
            status=EventStatus.ERROR,
            error_type="TimeoutError",
        )
    ]
    report = diff_events(left, right)
    assert report.n_status_changed == 1
    assert report.n_input_changed == 0
    assert report.n_output_changed == 0
    assert report.changes[0].kind is ChangeKind.STATUS_CHANGED


def test_deleted_unaligned_left():
    left = [
        make_event(1, call_site=CS_P),
        make_event(2, call_site=CS_T, seam=SeamKind.TOOL),
    ]
    right = [make_event(1, call_site=CS_P)]
    report = diff_events(left, right)
    assert report.n_deleted == 1
    assert report.n_inserted == 0
    assert report.changes[0].kind is ChangeKind.DELETED
    assert report.changes[0].left_seq == 2


def test_inserted_unaligned_right():
    left = [make_event(1, call_site=CS_P)]
    right = [
        make_event(1, call_site=CS_P),
        make_event(2, call_site=CS_T, seam=SeamKind.TOOL),
    ]
    report = diff_events(left, right)
    assert report.n_inserted == 1
    assert report.n_deleted == 0
    assert report.changes[0].kind is ChangeKind.INSERTED
    assert report.changes[0].right_seq == 2


def test_positional_shift_alone_is_unchanged():
    """seq/ordinal/lamport drift after an insert is not a semantic change."""
    digest = "b2b:" + "a" * 64
    left = [make_event(5, ordinal=0, lamport=5, input_digest=digest)]
    right = [make_event(9, ordinal=3, lamport=9, input_digest=digest)]
    report = diff_events(left, right)
    assert report.identical
    assert report.n_unchanged == 1


# ── Seq 0 and length mismatches ──────────────────────────────────────────────


def test_divergence_at_seq_zero():
    left = [make_event(0, ordinal=0, input_digest="b2b:" + "a" * 64)]
    right = [make_event(0, ordinal=0, input_digest="b2b:" + "b" * 64)]
    report = diff_events(left, right)
    assert report.first_divergence is not None
    assert report.first_divergence.seq == 0
    assert "seq=0" in report.first_divergence.message
    assert "no common prefix" in report.first_divergence.message


def test_different_lengths_left_longer():
    left = [
        make_event(1, call_site=CS_P, ordinal=0),
        make_event(2, call_site=CS_T, seam=SeamKind.TOOL, ordinal=0),
        make_event(3, call_site="site-extra", ordinal=0),
    ]
    right = [
        make_event(1, call_site=CS_P, ordinal=0),
        make_event(2, call_site=CS_T, seam=SeamKind.TOOL, ordinal=0),
    ]
    report = diff_events(left, right)
    assert report.n_deleted == 1
    assert report.n_inserted == 0
    assert report.n_unchanged == 2
    assert report.changes[0].kind is ChangeKind.DELETED


def test_different_lengths_right_longer():
    left = [
        make_event(1, call_site=CS_P, ordinal=0),
        make_event(2, call_site=CS_T, seam=SeamKind.TOOL, ordinal=0),
    ]
    right = [
        make_event(1, call_site=CS_P, ordinal=0),
        make_event(2, call_site=CS_T, seam=SeamKind.TOOL, ordinal=0),
        make_event(3, call_site="site-extra", ordinal=0),
    ]
    report = diff_events(left, right)
    assert report.n_inserted == 1
    assert report.n_deleted == 0
    assert report.n_unchanged == 2
    assert report.changes[0].kind is ChangeKind.INSERTED


def test_message_names_last_common_event_exactly():
    left = [
        make_event(
            40,
            call_site=CS_T,
            seam=SeamKind.TOOL,
            ordinal=0,
            input_digest="b2b:" + "t" * 64,
        ),
        make_event(41, ordinal=0, input_digest="b2b:" + "a" * 64),
    ]
    right = [
        make_event(
            40,
            call_site=CS_T,
            seam=SeamKind.TOOL,
            ordinal=0,
            input_digest="b2b:" + "t" * 64,
        ),
        make_event(41, ordinal=0, input_digest="b2b:" + "b" * 64),
    ]
    report = diff_events(left, right)
    assert report.first_divergence is not None
    assert report.first_divergence.seq == 41
    assert report.first_divergence.last_common_seq == 40
    assert "Last common event: seq=40" in report.first_divergence.message
    assert "at or before" not in report.first_divergence.message


# ── Agents ───────────────────────────────────────────────────────────────────


def test_per_agent_alignment_does_not_scramble_the_other_agent():
    left = [
        make_event(
            1,
            agent_id="root",
            call_site=CS_P,
            ordinal=0,
            input_digest="b2b:" + "a" * 64,
        ),
        make_event(
            2,
            agent_id="worker",
            call_site=CS_T,
            seam=SeamKind.TOOL,
            ordinal=0,
            input_digest="b2b:" + "w" * 64,
        ),
        make_event(
            3,
            agent_id="root",
            call_site=CS_P,
            ordinal=1,
            input_digest="b2b:" + "b" * 64,
        ),
    ]
    right = [
        make_event(
            1,
            agent_id="root",
            call_site=CS_P,
            ordinal=0,
            input_digest="b2b:" + "a" * 64,
        ),
        make_event(
            2,
            agent_id="worker",
            call_site=CS_T,
            seam=SeamKind.TOOL,
            ordinal=0,
            input_digest="b2b:" + "w" * 64,
        ),
        make_event(
            3,
            agent_id="worker",
            call_site=CS_T,
            seam=SeamKind.TOOL,
            ordinal=1,
            input_digest="b2b:" + "x" * 64,
        ),
        make_event(
            4,
            agent_id="root",
            call_site=CS_P,
            ordinal=1,
            input_digest="b2b:" + "b" * 64,
        ),
    ]
    report = diff_events(left, right)
    assert report.n_unchanged == 3
    assert report.n_inserted == 1
    assert report.n_input_changed == 0
    assert report.changes[0].agent_id == "worker"


def test_agent_id_set_difference_is_reported_not_dropped():
    left = [make_event(1, agent_id="root", ordinal=0)]
    right = [
        make_event(1, agent_id="root", ordinal=0),
        make_event(
            2, agent_id="worker", ordinal=0, call_site=CS_T, seam=SeamKind.TOOL
        ),
    ]
    report = diff_events(left, right)
    assert report.agent_ids_differ
    assert report.left_agent_ids == ("root",)
    assert report.right_agent_ids == ("root", "worker")
    assert report.n_inserted == 1
    assert report.changes[0].agent_id == "worker"
    assert report.first_divergence is not None
    assert "Agent ids differ" in report.first_divergence.message


def test_replace_of_a_different_call_site_is_delete_plus_insert():
    left = [
        make_event(1, call_site=CS_P, ordinal=0),
        make_event(2, call_site=CS_T, seam=SeamKind.TOOL, ordinal=0),
        make_event(3, call_site=CS_P, ordinal=1),
    ]
    right = [
        make_event(1, call_site=CS_P, ordinal=0),
        make_event(2, call_site="site-other", ordinal=0),
        make_event(3, call_site=CS_P, ordinal=1),
    ]
    report = diff_events(left, right)
    kinds = {c.kind for c in report.changes}
    assert kinds == {ChangeKind.DELETED, ChangeKind.INSERTED}
    assert report.n_input_changed == 0


# ── JSON contract ────────────────────────────────────────────────────────────


def test_diff_report_json_round_trips_into_the_dataclass():
    left = [make_event(0, ordinal=0, input_digest="b2b:" + "a" * 64)]
    right = [make_event(0, ordinal=0, input_digest="b2b:" + "b" * 64)]
    report = diff_events(left, right)
    payload = json.loads(json.dumps(report.to_dict()))
    restored = DiffReport.from_dict(payload)
    assert restored == report
    assert restored.first_divergence is not None
    assert restored.first_divergence.kind is ChangeKind.INPUT_CHANGED


def test_from_dict_rejects_a_truncated_payload():
    with pytest.raises(ValueError, match="missing field"):
        DiffReport.from_dict({"identical": True})


# ── Guards ───────────────────────────────────────────────────────────────────


def test_compare_does_not_load_blobs(tmp_path, monkeypatch):
    path = _write_trace(tmp_path, [make_event(1, ordinal=0)])

    def boom(*_a, **_k):
        raise AssertionError("diff must not load blobs during comparison")

    monkeypatch.setattr(BlobStore, "get", boom)
    monkeypatch.setattr(BlobStore, "get_obj", boom)
    report = compare_paths(path, path)
    assert report.identical


def test_seam_codec_mismatch_is_incomparable(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(
        tmp_path / "a", events, header_kwargs={"seam_codecs": {"provider": "fp-a"}}
    )
    right = _write_trace(
        tmp_path / "b", events, header_kwargs={"seam_codecs": {"provider": "fp-b"}}
    )
    with pytest.raises(IncomparableError, match="seam codec fingerprints differ"):
        compare_paths(left, right)


def test_tainted_trace_is_incomparable(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(tmp_path / "a", events, header_kwargs={"policy": "strict"})
    right = _write_trace(tmp_path / "b", events, header_kwargs={"policy": "lenient"})
    with pytest.raises(IncomparableError, match="tainted"):
        compare_paths(left, right)


def test_tainted_event_status_is_incomparable(tmp_path):
    good = [make_event(1, ordinal=0)]
    bad = [make_event(1, ordinal=0, status=EventStatus.TAINTED)]
    left = _write_trace(tmp_path / "a", good)
    right = _write_trace(tmp_path / "b", bad)
    with pytest.raises(IncomparableError, match="tainted"):
        compare_paths(left, right)


def test_schema_major_mismatch_is_incomparable(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(tmp_path / "a", events)
    right = _write_trace(tmp_path / "b", events)
    lines = right.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["schema_version"] = "9.0.0"
    lines[0] = json.dumps(header)
    right.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IncomparableError, match="schema major versions differ"):
        compare_paths(left, right)


def test_missing_trace_is_incomparable(tmp_path):
    existing = _write_trace(tmp_path, [make_event(1, ordinal=0)])
    with pytest.raises(IncomparableError, match="missing"):
        compare_paths(existing, tmp_path / "nope.jsonl")


def test_unreadable_trace_is_incomparable(tmp_path):
    existing = _write_trace(tmp_path, [make_event(1, ordinal=0)])
    garbage = tmp_path / "garbage.jsonl"
    garbage.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(IncomparableError, match="unreadable"):
        compare_paths(existing, garbage)


def test_empty_file_is_unreadable(tmp_path):
    existing = _write_trace(tmp_path, [make_event(1, ordinal=0)])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(IncomparableError, match="no header"):
        compare_paths(existing, empty)


def test_git_sha_mismatch_warns_but_proceeds(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(
        tmp_path / "a", events, header_kwargs={"git_sha": "a" * 40}
    )
    right = _write_trace(
        tmp_path / "b", events, header_kwargs={"git_sha": "b" * 40}
    )
    report = compare_paths(left, right)
    assert report.identical
    assert any("git sha differs" in w for w in report.warnings)


def test_schema_minor_mismatch_warns_but_proceeds(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(
        tmp_path / "a", events, header_kwargs={"schema_version": "0.2.0"}
    )
    right = _write_trace(
        tmp_path / "b", events, header_kwargs={"schema_version": SCHEMA_VERSION}
    )
    report = compare_paths(left, right)
    assert report.identical
    assert any("schema minor" in w for w in report.warnings)


def test_target_argv_mismatch_warns_loudly(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(
        tmp_path / "a", events, header_kwargs={"target": ["good.py"]}
    )
    right = _write_trace(
        tmp_path / "b", events, header_kwargs={"target": ["bad.py", "--flag"]}
    )
    report = compare_paths(left, right)
    assert report.identical
    blob = "\n".join(report.warnings)
    assert "target argv differs" in blob
    assert "usually a mistake" in blob
    assert "good.py" in blob
    assert "bad.py" in blob


# ── Human renderer ───────────────────────────────────────────────────────────


def test_render_human_identical():
    events = [make_event(1, ordinal=0)]
    text = render_human(diff_events(events, events), events, events)
    assert text.startswith("identical:")


def test_render_human_shows_context_and_unified_input_diff(tmp_path):
    left_input = {"prompt": "hello"}
    right_input = {"prompt": "HELLO"}
    left_digest = digest_obj(left_input)
    right_digest = digest_obj(right_input)
    left_events = [
        make_event(1, call_site=CS_T, seam=SeamKind.TOOL, ordinal=0),
        make_event(2, ordinal=0, input_digest=left_digest),
    ]
    right_events = [
        make_event(1, call_site=CS_T, seam=SeamKind.TOOL, ordinal=0),
        make_event(2, ordinal=0, input_digest=right_digest),
    ]
    left_path = _write_trace(
        tmp_path / "a", left_events, inputs={"l": left_input}
    )
    right_path = _write_trace(
        tmp_path / "b", right_events, inputs={"r": right_input}
    )
    left_reader = TraceReader(left_path)
    right_reader = TraceReader(right_path)
    report = diff_events(left_reader.events, right_reader.events)
    text = render_human(
        report,
        left_reader.events,
        right_reader.events,
        context=1,
        left_blobs=left_reader.blobs,
        right_blobs=right_reader.blobs,
        left_stores_inputs=True,
        right_stores_inputs=True,
    )
    assert "First divergence at seq=2" in text
    assert "Last common event: seq=1" in text
    assert "hello" in text
    assert "HELLO" in text
    assert "--- good seq=2" in text
    assert "+++ bad seq=2" in text


def test_render_human_names_missing_input_blobs():
    left = [make_event(0, ordinal=0, input_digest="b2b:" + "a" * 64)]
    right = [make_event(0, ordinal=0, input_digest="b2b:" + "b" * 64)]
    text = render_human(diff_events(left, right), left, right, context=0)
    assert "input blobs are not in the store" in text


def test_identity_key_excludes_ordinal_and_digest():
    a = make_event(1, ordinal=0, input_digest="b2b:" + "a" * 64)
    b = make_event(9, ordinal=7, input_digest="b2b:" + "b" * 64)
    assert identity_key(a) == identity_key(b)
    assert identity_key(a) == (SeamKind.PROVIDER.value, CS_P, "root")


def test_compare_readers_round_trip_through_trace_writer(tmp_path):
    events = [
        make_event(1, ordinal=0),
        make_event(2, call_site=CS_T, seam=SeamKind.TOOL),
    ]
    path = _write_trace(
        tmp_path, events, header_kwargs={"stores_input_blobs": True}
    )
    reader = TraceReader(path)
    report = compare_readers(reader, reader)
    assert report.identical
    assert report.warnings == ()


def test_identity_redactor_does_not_store_input_blobs_by_default(tmp_path):
    """The M0 redactor is identity. Storing inputs would write raw prompts."""
    cs = call_site_id(__name__, "echo", SeamKind.PROVIDER)
    header = RunHeader.new()
    payload = {"prompt": "must not land on disk"}
    with TraceWriter(tmp_path, header) as writer:
        rec = Recorder(writer)
        with use_interceptor(rec):
            intercept(SeamKind.PROVIDER, cs, payload, lambda: {"text": "ok"})
    store = BlobStore(tmp_path)
    digest = rec.events[0].input_digest
    assert digest == digest_obj(payload)
    assert not store.has(digest)
    assert rec.events[0].output_ref is not None
    assert store.has(rec.events[0].output_ref)


def test_store_inputs_writes_the_payload_under_its_digest(tmp_path):
    cs = call_site_id(__name__, "echo", SeamKind.PROVIDER)
    header = RunHeader.new(stores_input_blobs=True)
    payload = {"prompt": "store me"}
    with TraceWriter(tmp_path, header) as writer:
        rec = Recorder(writer, store_inputs=True)
        with use_interceptor(rec):
            intercept(SeamKind.PROVIDER, cs, payload, lambda: {"text": "ok"})
    store = BlobStore(tmp_path)
    assert rec.events[0].input_digest == digest_obj(payload)
    assert store.has(rec.events[0].input_digest)
    assert store.get_obj(rec.events[0].input_digest) == payload


def test_render_human_inserted_side_has_no_left_locus():
    left = [make_event(1, call_site=CS_P, ordinal=0)]
    right = [
        make_event(1, call_site="site-new", ordinal=0),
        make_event(2, call_site=CS_P, ordinal=0),
    ]
    text = render_human(diff_events(left, right), left, right, context=1)
    assert "First divergence" in text
    assert "inserted" in text


def test_from_dict_change_and_divergence_missing_fields():
    from agentos.replay.diff import Change, Divergence

    with pytest.raises(ValueError, match="missing field"):
        Change.from_dict({"kind": "inserted"})
    with pytest.raises(ValueError, match="missing field"):
        Divergence.from_dict({"seq": 0})


def test_first_record_not_a_header_is_unreadable(tmp_path):
    existing = _write_trace(tmp_path, [make_event(1, ordinal=0)])
    bad = tmp_path / "noheader.jsonl"
    bad.write_text('{"record_type":"event","seq":1}\n', encoding="utf-8")
    with pytest.raises(IncomparableError, match="not a header"):
        compare_paths(existing, bad)


def test_first_record_not_an_object_is_unreadable(tmp_path):
    existing = _write_trace(tmp_path, [make_event(1, ordinal=0)])
    bad = tmp_path / "array.jsonl"
    bad.write_text("[]\n", encoding="utf-8")
    with pytest.raises(IncomparableError, match="not an object"):
        compare_paths(existing, bad)


def test_malformed_event_is_unreadable(tmp_path):
    good = _write_trace(tmp_path / "a", [make_event(1, ordinal=0)])
    bad = _write_trace(tmp_path / "b", [make_event(1, ordinal=0)])
    with bad.open("a", encoding="utf-8") as fh:
        fh.write('{"record_type":"event","seq":99}\n')
    with pytest.raises(IncomparableError, match="unreadable"):
        compare_paths(good, bad)


def test_shared_future_major_is_unreadable(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(tmp_path / "a", events)
    right = _write_trace(tmp_path / "b", events)
    for path in (left, right):
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        header["schema_version"] = "9.0.0"
        lines[0] = json.dumps(header)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IncomparableError, match="not readable"):
        compare_paths(left, right)


def test_compare_readers_refuses_major_mismatch(tmp_path):
    from dataclasses import replace

    path = _write_trace(tmp_path, [make_event(1, ordinal=0)])
    left = TraceReader(path)
    right = TraceReader(path)
    right.header = replace(right.header, schema_version="9.0.0")
    with pytest.raises(IncomparableError, match="schema major"):
        compare_readers(left, right)


def test_render_empty_left_side():
    right = [make_event(0, call_site="only-right", ordinal=0)]
    text = render_human(diff_events([], right), [], right, context=0)
    assert "(no events)" in text
    assert "inserted" in text


def test_render_clamps_negative_context():
    left = [make_event(0, ordinal=0, input_digest="b2b:" + "a" * 64)]
    right = [make_event(0, ordinal=0, input_digest="b2b:" + "b" * 64)]
    text = render_human(diff_events(left, right), left, right, context=-3)
    assert "First divergence at seq=0" in text


def test_render_identical_inputs_at_output_change(tmp_path):
    payload = {"prompt": "same"}
    digest = digest_obj(payload)
    left = [
        make_event(1, ordinal=0, input_digest=digest, output_ref="b2b:" + "1" * 64)
    ]
    right = [
        make_event(1, ordinal=0, input_digest=digest, output_ref="b2b:" + "2" * 64)
    ]
    _write_trace(tmp_path, left, inputs={"p": payload})
    store = BlobStore(tmp_path)
    text = render_human(
        diff_events(left, right),
        left,
        right,
        left_blobs=store,
        right_blobs=store,
        left_stores_inputs=True,
        right_stores_inputs=True,
    )
    assert "canonical inputs are identical" in text


def test_missing_left_trace(tmp_path):
    right = _write_trace(tmp_path, [make_event(1, ordinal=0)])
    with pytest.raises(IncomparableError, match="left trace is missing"):
        compare_paths(tmp_path / "ghost.jsonl", right)


def test_target_none_vs_empty_warns(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(tmp_path / "a", events, header_kwargs={"target": None})
    right = _write_trace(tmp_path / "b", events, header_kwargs={"target": []})
    report = compare_paths(left, right)
    assert any("target argv differs" in w for w in report.warnings)


def test_header_without_schema_version_defaults_major_to_zero(tmp_path):
    events = [make_event(1, ordinal=0)]
    left = _write_trace(
        tmp_path / "a", events, header_kwargs={"schema_version": "0.2.0"}
    )
    right = tmp_path / "b" / "runs" / "x.jsonl"
    right.parent.mkdir(parents=True)
    header = json.loads(left.read_text(encoding="utf-8").splitlines()[0])
    header.pop("schema_version", None)
    event = json.loads(left.read_text(encoding="utf-8").splitlines()[1])
    right.write_text(
        json.dumps(header) + "\n" + json.dumps(event) + "\n", encoding="utf-8"
    )
    report = compare_paths(left, right)
    assert any("schema minor" in w for w in report.warnings)


def test_render_empty_right_side():
    left = [make_event(0, ordinal=0)]
    text = render_human(diff_events(left, []), left, [], context=0)
    assert "(no events)" in text
    assert "deleted" in text


def test_render_with_store_but_missing_blobs(tmp_path):
    left = [make_event(0, ordinal=0, input_digest="b2b:" + "a" * 64)]
    right = [make_event(0, ordinal=0, input_digest="b2b:" + "b" * 64)]
    store = BlobStore(tmp_path)
    text = render_human(
        diff_events(left, right),
        left,
        right,
        left_blobs=store,
        right_blobs=store,
    )
    assert "input blobs are not in the store" in text


def test_header_allows_leading_blank_lines(tmp_path):
    events = [make_event(1, ordinal=0)]
    path = _write_trace(tmp_path / "a", events)
    raw = path.read_text(encoding="utf-8")
    path.write_text("\n\n" + raw, encoding="utf-8")
    report = compare_paths(path, path)
    assert report.identical


def _as_legacy_0_3_0(path: Path) -> Path:
    """A 0.3.0 header has input_digests and no stores_input_blobs field."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["schema_version"] = "0.3.0"
    header.pop("stores_input_blobs", None)
    lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_v0_3_0_trace_diffs_on_digests_and_is_not_incomparable(tmp_path):
    """A missing input blob on a 0.3.0 trace is expected, not a KeyError/125."""
    digest_a = "b2b:" + "a" * 64
    digest_b = "b2b:" + "b" * 64
    left = _as_legacy_0_3_0(
        _write_trace(
            tmp_path / "a",
            [make_event(1, ordinal=0, input_digest=digest_a)],
        )
    )
    right = _as_legacy_0_3_0(
        _write_trace(
            tmp_path / "b",
            [make_event(1, ordinal=0, input_digest=digest_b)],
        )
    )
    report = compare_paths(left, right)
    assert not report.identical
    assert report.n_input_changed == 1
    assert any(
        w == "left trace predates input storage; comparing digests only"
        for w in report.warnings
    )
    assert any(
        w == "right trace predates input storage; comparing digests only"
        for w in report.warnings
    )
    left_reader = TraceReader(left)
    right_reader = TraceReader(right)
    assert left_reader.header.stores_input_blobs is False
    assert left_reader.header.schema_version == "0.3.0"
    text = render_human(
        report,
        left_reader.events,
        right_reader.events,
        left_blobs=left_reader.blobs,
        right_blobs=right_reader.blobs,
        left_stores_inputs=left_reader.header.stores_input_blobs,
        right_stores_inputs=right_reader.header.stores_input_blobs,
    )
    assert "input blobs are not in the store" in text


def test_v0_3_0_self_diff_is_identical_with_predate_warning(tmp_path):
    path = _as_legacy_0_3_0(
        _write_trace(tmp_path, [make_event(1, ordinal=0)])
    )
    report = compare_paths(path, path)
    assert report.identical
    assert "left trace predates input storage; comparing digests only" in report.warnings

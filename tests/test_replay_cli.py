"""CLI surface for record and replay.

The exit codes are the contract here, not the output. `git bisect run` shells
out to `agentos replay` and decides "good", "bad", or "skip" purely from the
status, so every code below is asserted directly rather than inferred from a
message. A wrong code does not produce a visible error, it produces a
confident wrong bisect result, which is far more expensive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentos import cli
from agentos.providers import mock as mock_mod

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reads its prompt and call count from the environment, so a replay can be made
# to diverge in the two distinct ways that matter: same call sites with
# different inputs, and more call sites than were recorded.
TARGET_SRC = """
import os

from agentos.providers.router import call_model

prompt = os.environ.get("AGENTOS_CLI_TEST_PROMPT", "first")
calls = int(os.environ.get("AGENTOS_CLI_TEST_CALLS", "1"))

for i in range(calls):
    call_model(
        "gpt-4o-mini",
        [{"role": "user", "content": f"{prompt}-{i}"}],
        [],
        agent_name="cli-test",
    )
"""


def run_cli(*argv: str) -> int:
    """Normalize the two ways the CLI can report a code."""
    try:
        return cli.main(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0


def only_run_id(trace_dir: Path) -> str:
    traces = list((trace_dir / "runs").glob("*.jsonl"))
    assert len(traces) == 1, f"expected one trace, found {traces}"
    return traces[0].stem


def read_header(trace_dir: Path, run_id: str) -> dict:
    path = trace_dir / "runs" / f"{run_id}.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def patch_header(trace_dir: Path, run_id: str, **changes) -> None:
    """Rewrite fields in a trace's header line.

    Tampering with a real recording beats hand-building a fake one: the rest of
    the file stays genuine, so the refusal under test is the only difference.
    """
    path = trace_dir / "runs" / f"{run_id}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header.update(changes)
    lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    monkeypatch.setenv("AGENTOS_DEMO_MODE", "true")
    monkeypatch.delenv("AGENTOS_CLI_TEST_PROMPT", raising=False)
    monkeypatch.delenv("AGENTOS_CLI_TEST_CALLS", raising=False)


@pytest.fixture
def live_calls(monkeypatch):
    counter = {"n": 0}
    real = mock_mod.call_mock

    def counting(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mock_mod, "call_mock", counting)
    return counter


@pytest.fixture
def target(tmp_path) -> Path:
    script = tmp_path / "cli_target.py"
    script.write_text(TARGET_SRC, encoding="utf-8")
    return script


@pytest.fixture
def recorded(tmp_path, target) -> tuple[Path, str]:
    trace_dir = tmp_path / "traces"
    assert run_cli("record", "--trace-dir", str(trace_dir), "--", str(target)) == 0
    return trace_dir, only_run_id(trace_dir)


# ── Exit code 0: equivalent ──────────────────────────────────────────────────


def test_replay_of_an_untouched_recording_exits_zero(recorded, live_calls):
    trace_dir, run_id = recorded
    before = live_calls["n"]
    assert run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_OK
    assert live_calls["n"] == before, "replay must make zero live calls"


def test_record_writes_a_trace_with_provenance(recorded):
    trace_dir, run_id = recorded
    header = read_header(trace_dir, run_id)
    assert header["record_type"] == "header"
    assert header["schema_version"] == "0.4.0"
    assert header["stores_input_blobs"] is False
    assert header["seam_codecs"], "record must stamp the seam codec fingerprint"
    # Execution identity is a typed field, not free-form metadata.
    assert header["target"], "replay needs to know what was run"
    assert "target_argv" not in header["labels"]
    # git_sha is what the drift check compares against; None would make the
    # check permanently vacuous.
    assert header["git_sha"], "record must capture the commit it ran at"
    assert isinstance(header["git_dirty"], bool), "dirty must be recorded, not implied"


def test_labels_stay_free_form_and_separate_from_target(tmp_path, target):
    trace_dir = tmp_path / "traces"
    assert (
        run_cli(
            "record",
            "--trace-dir",
            str(trace_dir),
            "--label",
            "suite=m2",
            "--label",
            "target=not-the-real-one",
            "--",
            str(target),
        )
        == 0
    )
    header = read_header(trace_dir, only_run_id(trace_dir))
    # A caller squatting on "target" in labels must not affect what replay runs.
    assert header["labels"] == {"suite": "m2", "target": "not-the-real-one"}
    assert header["target"][0].endswith("cli_target.py")


# ── Exit code 2: divergence ──────────────────────────────────────────────────


def test_changed_input_exits_two_and_points_upstream(recorded, monkeypatch, capsys):
    trace_dir, run_id = recorded
    monkeypatch.setenv("AGENTOS_CLI_TEST_PROMPT", "second")

    code = run_cli("replay", run_id, "--trace-dir", str(trace_dir))
    assert code == cli.EXIT_DIVERGENCE

    err = capsys.readouterr().err
    assert "DIVERGENCE" in err
    assert "seq:" in err
    assert "call site:" in err
    assert "root cause is at or before seq=" in err


def test_extra_call_site_exits_two(recorded, monkeypatch, capsys):
    """Control flow ran past the recording, so there is no event to name."""
    trace_dir, run_id = recorded
    monkeypatch.setenv("AGENTOS_CLI_TEST_CALLS", "2")

    assert (
        run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_DIVERGENCE
    )
    err = capsys.readouterr().err
    assert "unrecorded" in err
    assert "root cause is at or before seq=" in err


# ── Exit code 125: untestable, git bisect must SKIP ──────────────────────────


def test_missing_trace_is_untestable(tmp_path):
    assert (
        run_cli("replay", "nope", "--trace-dir", str(tmp_path)) == cli.EXIT_UNTESTABLE
    )


def test_schema_major_mismatch_is_untestable(recorded):
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, schema_version="9.0.0")
    assert (
        run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_UNTESTABLE
    )


def test_seam_codec_mismatch_is_untestable(recorded, capsys):
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, seam_codecs={"provider": "from-an-older-build"})
    assert (
        run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_UNTESTABLE
    )
    assert "codec mismatch" in capsys.readouterr().err


def test_git_drift_is_untestable_unless_allowed(recorded, capsys, live_calls):
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, git_sha="0" * 40)

    assert (
        run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_UNTESTABLE
    )
    assert "HEAD is" in capsys.readouterr().err

    before = live_calls["n"]
    assert (
        run_cli("replay", run_id, "--trace-dir", str(trace_dir), "--allow-drift")
        == cli.EXIT_OK
    )
    assert live_calls["n"] == before


def test_tainted_trace_is_refused(recorded, capsys):
    """A tainted run is not valid input for any comparison (DETERMINISM.md 5)."""
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, policy="lenient")
    assert (
        run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_UNTESTABLE
    )
    assert "tainted" in capsys.readouterr().err


def test_lenient_replay_that_falls_through_is_refused(recorded, monkeypatch, capsys):
    """LENIENT can still answer 0 or 2; a run it had to taint cannot."""
    trace_dir, run_id = recorded
    monkeypatch.setenv("AGENTOS_CLI_TEST_PROMPT", "second")
    code = run_cli(
        "replay", run_id, "--trace-dir", str(trace_dir), "--policy", "lenient"
    )
    assert code == cli.EXIT_UNTESTABLE
    assert "tainted" in capsys.readouterr().err


def test_non_python_target_is_untestable(tmp_path, capsys):
    code = run_cli("record", "--trace-dir", str(tmp_path), "--", "npm", "test")
    assert code == cli.EXIT_UNTESTABLE
    assert "in-process" in capsys.readouterr().err


def test_missing_target_file_is_untestable(tmp_path):
    code = run_cli(
        "record", "--trace-dir", str(tmp_path), "--", str(tmp_path / "ghost.py")
    )
    assert code == cli.EXIT_UNTESTABLE


def test_usage_error_is_untestable_not_divergence(tmp_path):
    """argparse exits 2 by default, which bisect would read as 'bad'."""
    assert run_cli("replay", "--no-such-flag") == cli.EXIT_UNTESTABLE


def test_pre_0_3_0_trace_without_target_is_untestable(recorded, capsys):
    """A 0.2.0 trace still loads, but cannot be replayed: say which, and why."""
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, schema_version="0.2.0", target=None)

    assert (
        run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_UNTESTABLE
    )
    err = capsys.readouterr().err
    assert "records no target" in err
    assert "0.3.0" in err, "the message must name the version that changed"


def test_unrunnable_recorded_target_is_untestable(recorded, capsys):
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, target=["not-a-python-target"])
    assert (
        run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_UNTESTABLE
    )
    assert "not runnable" in capsys.readouterr().err


# ── Secrets and provenance ───────────────────────────────────────────────────


def test_secret_flags_are_redacted_out_of_the_recorded_target(tmp_path, target):
    """argv reaches disk, and traces get committed and shared."""
    trace_dir = tmp_path / "traces"
    assert (
        run_cli(
            "record",
            "--trace-dir",
            str(trace_dir),
            "--",
            str(target),
            "--api-key",
            "sk-live-must-not-persist",
            "--token=ghp_must_not_persist",
            "PASSWORD=must-not-persist",
            "--depth",
            "3",
        )
        == 0
    )
    raw = (trace_dir / "runs" / f"{only_run_id(trace_dir)}.jsonl").read_text()
    assert "must-not-persist" not in raw
    assert "must_not_persist" not in raw

    stored = read_header(trace_dir, only_run_id(trace_dir))["target"]
    assert stored[1:] == [
        "--api-key",
        cli.REDACTED,
        f"--token={cli.REDACTED}",
        f"PASSWORD={cli.REDACTED}",
        "--depth",
        "3",
    ], "only secret-bearing values are masked; ordinary args survive"


def test_replay_warns_when_the_target_was_redacted(tmp_path, target, capsys):
    trace_dir = tmp_path / "traces"
    run_cli(
        "record", "--trace-dir", str(trace_dir), "--", str(target), "--token", "shh"
    )
    capsys.readouterr()
    run_cli("replay", only_run_id(trace_dir), "--trace-dir", str(trace_dir))
    assert "redacted secret" in capsys.readouterr().err


def test_dirty_tree_warns_even_when_the_sha_matches(recorded, capsys):
    """A matching sha reads as 'same code'; a dirty tree means it is not."""
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, git_dirty=True)
    assert run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_OK
    err = capsys.readouterr().err
    assert "dirty tree" in err


def test_unknown_provenance_skips_the_drift_check(recorded, capsys, monkeypatch):
    """Outside a git repo the check must skip, not refuse."""
    monkeypatch.setattr(cli, "_git_provenance", lambda cwd: (None, False))
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, git_sha=None)

    assert run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_OK
    assert "drift check is skipped" in capsys.readouterr().err


def test_record_warns_when_provenance_is_unknown(tmp_path, target, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_git_provenance", lambda cwd: (None, False))
    trace_dir = tmp_path / "traces"
    assert run_cli("record", "--trace-dir", str(trace_dir), "--", str(target)) == 0

    assert "no git commit could be determined" in capsys.readouterr().err
    assert read_header(trace_dir, only_run_id(trace_dir))["git_sha"] is None


def test_record_warns_when_the_tree_is_dirty(tmp_path, target, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_git_provenance", lambda cwd: ("a" * 40, True))
    trace_dir = tmp_path / "traces"
    assert run_cli("record", "--trace-dir", str(trace_dir), "--", str(target)) == 0

    assert "tree is dirty" in capsys.readouterr().err
    assert read_header(trace_dir, only_run_id(trace_dir))["git_dirty"] is True


# ── Codes git bisect cannot survive ──────────────────────────────────────────


def test_no_invocation_can_exit_126_or_127(tmp_path, target, recorded):
    """`git bisect run` aborts the session on 126/127 instead of skipping."""
    trace_dir, run_id = recorded
    invocations = [
        (),
        ("--help-me-please",),
        ("nonsense",),
        ("trace",),
        ("trace", "ls", "--trace-dir", str(tmp_path)),
        ("trace", "show", "nope", "--trace-dir", str(tmp_path)),
        ("trace", "gc", "--trace-dir", str(tmp_path)),
        ("record", "--trace-dir", str(tmp_path)),
        ("record", "--trace-dir", str(tmp_path), "--", "npm", "test"),
        ("replay",),
        ("replay", "nope", "--trace-dir", str(tmp_path)),
        ("replay", run_id, "--trace-dir", str(trace_dir)),
        ("replay", run_id, "--trace-dir", str(trace_dir), "--policy", "bogus"),
        ("diff",),
        ("diff", "nope.jsonl", "nope.jsonl"),
        ("diff", run_id, run_id, "--trace-dir", str(trace_dir)),
        ("diff", run_id, run_id, "--trace-dir", str(trace_dir), "--json"),
    ]
    for argv in invocations:
        code = run_cli(*argv)
        assert code not in (126, 127), f"{argv} exited {code}"
        assert code in (0, 1, 2, 125), f"{argv} exited unexpected {code}"


def test_target_exit_code_cannot_masquerade_as_a_verdict(tmp_path):
    """A target exiting 127 must not become the CLI's 127."""
    script = tmp_path / "exits.py"
    script.write_text("import sys\nsys.exit(127)\n", encoding="utf-8")
    code = run_cli("record", "--trace-dir", str(tmp_path / "t"), "--", str(script))
    assert code == 1


def test_target_exiting_zero_is_success(tmp_path):
    script = tmp_path / "clean.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    assert run_cli("record", "--trace-dir", str(tmp_path / "t"), "--", str(script)) == 0


def test_target_that_raises_exits_one(tmp_path, capsys):
    """Target failure is bisect BAD (1), not SKIP (125)."""
    script = tmp_path / "boom.py"
    script.write_text("raise RuntimeError('target blew up')\n", encoding="utf-8")
    code = run_cli("record", "--trace-dir", str(tmp_path / "t"), "--", str(script))
    assert code == 1
    assert "target raised RuntimeError" in capsys.readouterr().err


def test_recorder_infrastructure_failure_exits_125(tmp_path, target, monkeypatch, capsys):
    """A bug in the tool must not look like a bad target to git bisect."""
    from agentos.replay.store import BlobStore

    def boom(self, obj):
        raise OSError("disk full")

    monkeypatch.setattr(BlobStore, "put_obj", boom)
    code = run_cli("record", "--trace-dir", str(tmp_path / "t"), "--", str(target))
    assert code == cli.EXIT_UNTESTABLE
    err = capsys.readouterr().err
    assert "recorder failed" in err
    assert "disk full" in err


def test_git_absent_degrades_to_none_sha_not_empty_string(
    tmp_path, target, monkeypatch
):
    """Failed git must yield None, never '' — empty sha would silence drift."""
    import subprocess as sp

    real_run = sp.run

    def fake_run(argv, *args, **kwargs):
        if argv and argv[0] == "git":
            # Explicit check=False path: nonzero + empty stdout → None upstream.
            return sp.CompletedProcess(argv, returncode=128, stdout="", stderr="fatal")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(sp, "run", fake_run)
    assert run_cli("record", "--trace-dir", str(tmp_path / "t"), "--", str(target)) == 0
    header = read_header(tmp_path / "t", only_run_id(tmp_path / "t"))
    assert header["git_sha"] is None
    assert header["git_sha"] != ""


# ── trace ls / show / gc ─────────────────────────────────────────────────────


def test_trace_ls_and_show(recorded, capsys):
    trace_dir, run_id = recorded

    assert run_cli("trace", "ls", "--trace-dir", str(trace_dir)) == cli.EXIT_OK
    assert run_id in capsys.readouterr().out

    assert (
        run_cli("trace", "show", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_OK
    )
    out = capsys.readouterr().out
    assert "provider" in out
    assert "seam codecs" in out


def test_trace_gc_collects_only_orphans(recorded, capsys):
    trace_dir, _run_id = recorded
    blobs = trace_dir / "blobs"
    referenced_before = {p.name for p in blobs.rglob("*.bin*")}

    orphan = blobs / "ab" / "cd" / "abcd0123.bin"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"unreferenced")

    assert (
        run_cli("trace", "gc", "--trace-dir", str(trace_dir), "--dry-run")
        == cli.EXIT_OK
    )
    assert orphan.exists(), "--dry-run must not delete"
    assert "would delete" in capsys.readouterr().out

    assert run_cli("trace", "gc", "--trace-dir", str(trace_dir)) == cli.EXIT_OK
    assert not orphan.exists()
    assert {p.name for p in blobs.rglob("*.bin*")} == referenced_before


def test_trace_gc_refuses_when_a_run_is_unreadable(recorded):
    """An unparseable run may reference blobs; deleting them would corrupt it."""
    trace_dir, run_id = recorded
    patch_header(trace_dir, run_id, schema_version="9.0.0")
    blobs_before = {p.name for p in (trace_dir / "blobs").rglob("*.bin*")}

    assert run_cli("trace", "gc", "--trace-dir", str(trace_dir)) == cli.EXIT_UNTESTABLE
    assert {p.name for p in (trace_dir / "blobs").rglob("*.bin*")} == blobs_before


# ── End to end against a real example ────────────────────────────────────────


def test_quickstart_records_and_replays_end_to_end(tmp_path, live_calls, monkeypatch):
    """The headline claim: a real example replays offline with zero spend."""
    monkeypatch.chdir(REPO_ROOT)  # quickstart.py does sys.path.insert(0, "src")
    trace_dir = tmp_path / "traces"

    assert (
        run_cli("record", "--trace-dir", str(trace_dir), "--", "examples/quickstart.py")
        == cli.EXIT_OK
    )
    recorded_calls = live_calls["n"]
    assert recorded_calls > 0, "recording a real example must hit the provider"

    run_id = only_run_id(trace_dir)
    assert run_cli("replay", run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_OK
    assert live_calls["n"] == recorded_calls, "replay must make zero live calls"


def test_record_honors_python_prefix_and_script_args(tmp_path, target):
    """`-- python script.py args` is what muscle memory types."""
    trace_dir = tmp_path / "traces"
    assert (
        run_cli(
            "record", "--trace-dir", str(trace_dir), "--", "python", str(target), "-x"
        )
        == cli.EXIT_OK
    )
    stored = read_header(trace_dir, only_run_id(trace_dir))["target"]
    # The interpreter token is normalized away, leaving script plus its args.
    assert len(stored) == 2, f"expected [script, -x], got {stored}"
    assert os.path.basename(stored[0]) == "cli_target.py"
    assert stored[-1] == "-x", "script args must survive for sys.argv"


# ── agentos diff ─────────────────────────────────────────────────────────────


def test_diff_of_a_trace_against_itself_exits_zero(recorded, capsys):
    trace_dir, run_id = recorded
    path = str(trace_dir / "runs" / f"{run_id}.jsonl")
    assert run_cli("diff", path, path) == cli.EXIT_OK
    assert "identical" in capsys.readouterr().out


def test_diff_json_round_trips_into_the_dataclass(recorded, capsys):
    from agentos.replay.diff import DiffReport

    trace_dir, run_id = recorded
    path = str(trace_dir / "runs" / f"{run_id}.jsonl")
    assert run_cli("diff", path, path, "--json") == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    report = DiffReport.from_dict(payload)
    assert report.identical
    assert report.first_divergence is None
    assert report.changes == [] or report.changes == ()


def test_diff_localizes_an_input_change(tmp_path, target, monkeypatch, capsys):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    assert run_cli("record", "--trace-dir", str(good_dir), "--", str(target)) == 0
    monkeypatch.setenv("AGENTOS_CLI_TEST_PROMPT", "second")
    assert run_cli("record", "--trace-dir", str(bad_dir), "--", str(target)) == 0
    capsys.readouterr()

    good = str(good_dir / "runs" / f"{only_run_id(good_dir)}.jsonl")
    bad = str(bad_dir / "runs" / f"{only_run_id(bad_dir)}.jsonl")
    assert run_cli("diff", good, bad, "--context", "1") == cli.EXIT_DIVERGENCE
    captured = capsys.readouterr()
    assert "First divergence at seq=" in captured.out
    assert "input_changed" in captured.out
    assert "at or before" not in captured.out
    # Default record does not store input blobs (identity redactor).
    assert "did not store input blobs" in captured.err
    assert "input blobs are not in the store" in captured.out


def test_diff_json_on_divergence_is_parseable(tmp_path, target, monkeypatch, capsys):
    from agentos.replay.diff import ChangeKind, DiffReport

    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    assert run_cli("record", "--trace-dir", str(good_dir), "--", str(target)) == 0
    monkeypatch.setenv("AGENTOS_CLI_TEST_PROMPT", "second")
    assert run_cli("record", "--trace-dir", str(bad_dir), "--", str(target)) == 0
    good = str(good_dir / "runs" / f"{only_run_id(good_dir)}.jsonl")
    bad = str(bad_dir / "runs" / f"{only_run_id(bad_dir)}.jsonl")
    capsys.readouterr()
    assert run_cli("diff", good, bad, "--json") == cli.EXIT_DIVERGENCE
    report = DiffReport.from_dict(json.loads(capsys.readouterr().out))
    assert not report.identical
    assert report.first_divergence is not None
    assert report.first_divergence.kind is ChangeKind.INPUT_CHANGED


def test_diff_missing_trace_is_untestable():
    assert (
        run_cli("diff", "nope-a.jsonl", "nope-b.jsonl") == cli.EXIT_UNTESTABLE
    )


def test_diff_missing_right_trace_is_untestable(recorded, capsys):
    trace_dir, run_id = recorded
    path = str(trace_dir / "runs" / f"{run_id}.jsonl")
    assert run_cli("diff", path, "nope-right.jsonl") == cli.EXIT_UNTESTABLE
    assert "no such trace" in capsys.readouterr().err


def test_diff_schema_major_mismatch_is_untestable(recorded, tmp_path):
    trace_dir, run_id = recorded
    good = trace_dir / "runs" / f"{run_id}.jsonl"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(good.read_text(encoding="utf-8"), encoding="utf-8")
    lines = bad.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["schema_version"] = "9.0.0"
    lines[0] = json.dumps(header)
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run_cli("diff", str(good), str(bad)) == cli.EXIT_UNTESTABLE


def test_diff_seam_codec_mismatch_is_untestable(recorded, capsys):
    trace_dir, run_id = recorded
    path = trace_dir / "runs" / f"{run_id}.jsonl"
    other = trace_dir / "runs" / "other.jsonl"
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    # Tamper the copy's codecs.
    lines = other.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["seam_codecs"] = {"provider": "from-an-older-build"}
    lines[0] = json.dumps(header)
    other.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run_cli("diff", str(path), str(other)) == cli.EXIT_UNTESTABLE
    assert "codec" in capsys.readouterr().err


def test_diff_tainted_trace_is_untestable(recorded, capsys):
    trace_dir, run_id = recorded
    path = str(trace_dir / "runs" / f"{run_id}.jsonl")
    patch_header(trace_dir, run_id, policy="lenient")
    assert run_cli("diff", path, path) == cli.EXIT_UNTESTABLE
    assert "tainted" in capsys.readouterr().err


def test_diff_unreadable_trace_is_untestable(recorded, tmp_path, capsys):
    trace_dir, run_id = recorded
    good = str(trace_dir / "runs" / f"{run_id}.jsonl")
    garbage = tmp_path / "garbage.jsonl"
    garbage.write_text("{not-json\n", encoding="utf-8")
    assert run_cli("diff", good, str(garbage)) == cli.EXIT_UNTESTABLE
    assert "unreadable" in capsys.readouterr().err


def test_diff_negative_context_is_untestable(recorded):
    trace_dir, run_id = recorded
    path = str(trace_dir / "runs" / f"{run_id}.jsonl")
    assert run_cli("diff", path, path, "--context", "-1") == cli.EXIT_UNTESTABLE


def test_diff_warns_on_git_sha_mismatch_but_proceeds(recorded, capsys):
    trace_dir, run_id = recorded
    path = trace_dir / "runs" / f"{run_id}.jsonl"
    other = trace_dir / "runs" / "other.jsonl"
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    lines = other.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["git_sha"] = "0" * 40
    lines[0] = json.dumps(header)
    other.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run_cli("diff", str(path), str(other)) == cli.EXIT_OK
    err = capsys.readouterr().err
    assert "git sha differs" in err
    assert "warning" in err


def test_diff_warns_on_target_argv_mismatch(recorded, capsys):
    trace_dir, run_id = recorded
    path = trace_dir / "runs" / f"{run_id}.jsonl"
    other = trace_dir / "runs" / "other.jsonl"
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    lines = other.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["target"] = ["other.py"]
    lines[0] = json.dumps(header)
    other.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run_cli("diff", str(path), str(other)) == cli.EXIT_OK
    err = capsys.readouterr().err
    assert "target argv differs" in err
    assert "usually a mistake" in err


def test_diff_accepts_run_ids_with_trace_dir(recorded, capsys):
    trace_dir, run_id = recorded
    assert (
        run_cli("diff", run_id, run_id, "--trace-dir", str(trace_dir)) == cli.EXIT_OK
    )
    assert "identical" in capsys.readouterr().out


def test_record_refuses_input_blobs_under_identity_redactor(
    tmp_path, target, capsys
):
    trace_dir = tmp_path / "traces"
    assert run_cli("record", "--trace-dir", str(trace_dir), "--", str(target)) == 0
    err = capsys.readouterr().err
    assert "not storing input blobs" in err
    assert 'redactor_version' in err
    assert "identity" in err

    header = read_header(trace_dir, only_run_id(trace_dir))
    assert header["redactor_version"] == "0"
    assert header["stores_input_blobs"] is False
    assert header["schema_version"] == "0.4.0"

    from agentos.replay import BlobStore, TraceReader

    reader = TraceReader(trace_dir / "runs" / f"{header['run_id']}.jsonl")
    store = BlobStore(trace_dir)
    assert reader.events, "the run must have recorded at least one seam"
    for event in reader.events:
        assert event.input_digest, "digests are recorded either way"
        assert not store.has(event.input_digest)


def test_store_inputs_unredacted_writes_blobs_and_says_why(
    tmp_path, target, capsys
):
    trace_dir = tmp_path / "traces"
    assert (
        run_cli(
            "record",
            "--trace-dir",
            str(trace_dir),
            "--store-inputs-unredacted",
            "--",
            str(target),
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "storing unredacted input blobs" in err
    assert "Do not commit" in err

    header = read_header(trace_dir, only_run_id(trace_dir))
    assert header["stores_input_blobs"] is True

    from agentos.replay import BlobStore, TraceReader

    reader = TraceReader(trace_dir / "runs" / f"{header['run_id']}.jsonl")
    store = BlobStore(trace_dir)
    assert any(store.has(event.input_digest) for event in reader.events)


def test_diff_legacy_0_3_0_exits_divergent_not_untestable(tmp_path, capsys):
    """A 0.3.0 header must not become exit 125 just because input blobs are absent."""
    from agentos.replay import RunHeader, TraceEvent, TraceWriter
    from agentos.replay.diff import ChangeKind, DiffReport
    from agentos.replay.schema import EventStatus, SeamKind

    def write(root, digest: str):
        header = RunHeader.new(schema_version="0.3.0")
        event = TraceEvent(
            event_id="e1",
            run_id=header.run_id,
            seq=1,
            seam=SeamKind.PROVIDER,
            call_site="site-p",
            ordinal=0,
            input_digest=digest,
            output_ref="b2b:" + "c" * 64,
            status=EventStatus.OK,
        )
        with TraceWriter(root, header) as writer:
            writer.append(event)
        path = root / "runs" / f"{header.run_id}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[0])
        rec["schema_version"] = "0.3.0"
        rec.pop("stores_input_blobs", None)
        lines[0] = json.dumps(rec)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    left = write(tmp_path / "a", "b2b:" + "a" * 64)
    right = write(tmp_path / "b", "b2b:" + "b" * 64)
    assert run_cli("diff", str(left), str(right), "--json") == cli.EXIT_DIVERGENCE
    captured = capsys.readouterr()
    assert "predates input storage" in captured.err
    report = DiffReport.from_dict(json.loads(captured.out))
    assert report.first_divergence is not None
    assert report.first_divergence.kind is ChangeKind.INPUT_CHANGED
"""`agentos bisect` git contract.

The product is one report: culprit commit AND first_divergence. These tests
drive a real throwaway git history, not a mock of `git bisect`, because the
SKIP-vs-BAD distinction only exists in git's exit-code decoder.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentos import cli
from agentos.providers import mock as mock_mod
from agentos.replay.bisect import (
    BisectSession,
    bisect_in_progress,
    env_is_bisect_mode,
    first_bad_ref,
    oracle_exit,
    rev_parse,
    run_git,
)


def run_cli(*argv: str) -> int:
    try:
        return cli.main(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0


def only_run_id(trace_dir: Path) -> str:
    traces = list((trace_dir / "runs").glob("*.jsonl"))
    assert len(traces) == 1, f"expected one trace, found {traces}"
    return traces[0].stem


def patch_header(trace_dir: Path, run_id: str, **changes) -> None:
    path = trace_dir / "runs" / f"{run_id}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header.update(changes)
    lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

TARGET_NAME = "target.py"
SKIP_IMPORT = "agentos_bisect_skip_module_that_does_not_exist"

_TARGET = """\
from agentos.providers.router import call_model
{extra}
PROMPT = "{prompt}"

if __name__ == "__main__":
    call_model(
        "gpt-4o-mini",
        [{{"role": "user", "content": PROMPT}}],
        [],
        agent_name="bisect-test",
    )
"""


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "AgentOS Bisect Test",
            "GIT_AUTHOR_EMAIL": "bisect@example.test",
            "GIT_COMMITTER_NAME": "AgentOS Bisect Test",
            "GIT_COMMITTER_EMAIL": "bisect@example.test",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "HUSKY": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "AGENTOS_DEMO_MODE": "true",
        }
    )
    env.pop("AGENTOS_BISECT", None)
    return env


def git_version() -> str:
    done = subprocess.run(
        ["git", "--version"], capture_output=True, text=True, check=False
    )
    return (done.stdout or done.stderr or "git --version produced no output").strip()


def bisect_marks(text: str) -> list[str]:
    """`git bisect good|bad|skip` lines from a captured `git bisect log`."""
    marks: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("git bisect ") and not stripped.startswith(
            "git bisect start"
        ):
            marks.append(stripped)
    return marks


def git(repo: Path, *argv: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(
        [
            "git",
            "-c",
            "user.name=AgentOS Bisect Test",
            "-c",
            "user.email=bisect@example.test",
            *argv,
        ],
        cwd=repo,
        env=_git_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and done.returncode != 0:
        raise AssertionError(
            f"git {argv} failed ({done.returncode}): {done.stderr or done.stdout}"
        )
    return done


def write_target(repo: Path, *, prompt: str, extra: str = "") -> None:
    (repo / TARGET_NAME).write_text(
        _TARGET.format(prompt=prompt, extra=extra), encoding="utf-8"
    )


def commit(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


@dataclass
class History:
    repo: Path
    good: str
    skip: str | None
    culprit: str
    head: str


def build_history(root: Path, *, with_skip: bool) -> History:
    """Linear history with >=10 commits in GOOD..BAD so git actually bisects.

    An earlier fixture jumped from n3 (good) to the culprit to n8–n10, so the
    search range was 4 commits and git reported "0 revisions left to test".
    That only proves git can compare two endpoints, not that the search
    converges. Skip sits at the first midpoint of the longer range so the
    SKIP-vs-BAD contract is actually exercised.
    """
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    write_target(repo, prompt="alpha")
    (repo / "notes.txt").write_text("n0\n", encoding="utf-8")
    commit(repo, "n0 initial alpha")

    for i in range(1, 3):
        (repo / "notes.txt").write_text(f"n{i}\n", encoding="utf-8")
        commit(repo, f"n{i} still alpha")
    good = git(repo, "rev-parse", "HEAD").stdout.strip()

    # Unknown-good half: n3–n7 still match the pre-culprit recording.
    for i in range(3, 8):
        (repo / "notes.txt").write_text(f"n{i}\n", encoding="utf-8")
        commit(repo, f"n{i} still alpha")

    skip_sha: str | None = None
    if with_skip:
        # 12-commit GOOD..BAD range; first midpoint is this skip commit.
        write_target(
            repo,
            prompt="alpha",
            extra=f"import {SKIP_IMPORT}  # untestable\n",
        )
        (repo / "notes.txt").write_text("skip\n", encoding="utf-8")
        skip_sha = commit(repo, "skip: package fails to import")
        write_target(repo, prompt="alpha")
        (repo / "notes.txt").write_text("after-skip\n", encoding="utf-8")
        commit(repo, "restore import, still alpha")

    write_target(repo, prompt="beta")
    (repo / "notes.txt").write_text("culprit\n", encoding="utf-8")
    culprit = commit(repo, "culprit: prompt becomes beta")

    for i in range(8, 12):
        (repo / "notes.txt").write_text(f"n{i}\n", encoding="utf-8")
        commit(repo, f"n{i} still beta")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    n_range = int(
        git(repo, "rev-list", "--count", f"{good}..{head}").stdout.strip()
    )
    if n_range < 10:
        raise AssertionError(
            f"GOOD..HEAD is {n_range} commits; need >= 10 so bisect searches "
            f"({git_version()})"
        )
    return History(repo=repo, good=good, skip=skip_sha, culprit=culprit, head=head)


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    monkeypatch.setenv("AGENTOS_DEMO_MODE", "true")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.delenv("AGENTOS_BISECT", raising=False)
    monkeypatch.delenv("AGENTOS_CLI_TEST_PROMPT", raising=False)


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
def history(tmp_path):
    h = build_history(tmp_path, with_skip=False)
    yield h
    git(h.repo, "bisect", "reset", check=False)
    git(h.repo, "checkout", "--quiet", "-f", h.head, check=False)


@pytest.fixture
def history_with_skip(tmp_path):
    h = build_history(tmp_path, with_skip=True)
    yield h
    git(h.repo, "bisect", "reset", check=False)
    git(h.repo, "checkout", "--quiet", "-f", h.head, check=False)


def record_at_tip(history: History, trace_dir: Path, monkeypatch) -> str:
    monkeypatch.chdir(history.repo)
    assert (
        run_cli(
            "record",
            "--trace-dir",
            str(trace_dir),
            "--",
            str(history.repo / TARGET_NAME),
        )
        == 0
    )
    return only_run_id(trace_dir)


def assert_head_restored(history: History) -> None:
    assert rev_parse(history.repo, "HEAD") == history.head
    assert not bisect_in_progress(history.repo)


# ── Oracle mapping ───────────────────────────────────────────────────────────


def test_oracle_exit_inverts_only_the_replay_verdicts():
    assert oracle_exit(cli.EXIT_OK) == 1
    assert oracle_exit(cli.EXIT_DIVERGENCE) == cli.EXIT_OK
    assert oracle_exit(cli.EXIT_UNTESTABLE) == cli.EXIT_UNTESTABLE
    assert oracle_exit(1) == 1


def test_first_bad_ref_is_populated_at_start_so_run_exit_is_the_verdict(history):
    """refs/bisect/bad names the original --bad from start; that is not a result."""
    session = BisectSession(repo=history.repo)
    session.capture_head()
    session.start(history.head, history.good)
    try:
        assert first_bad_ref(history.repo) == history.head
    finally:
        session.restore()
    assert_head_restored(history)


def test_env_is_bisect_mode(monkeypatch):
    monkeypatch.delenv("AGENTOS_BISECT", raising=False)
    assert env_is_bisect_mode() is False
    monkeypatch.setenv("AGENTOS_BISECT", "oracle")
    assert env_is_bisect_mode() is True


# ── Up-front refusals, exit 125, HEAD untouched ──────────────────────────────


def test_dirty_tree_is_refused_before_bisect_starts(
    history, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(history.repo)
    run_id = record_at_tip(history, tmp_path / "traces", monkeypatch)
    (history.repo / "unstaged.txt").write_text("dirt\n", encoding="utf-8")
    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(tmp_path / "traces"),
        "--good",
        history.good,
        "--no-diff",
    )
    assert code == cli.EXIT_UNTESTABLE
    assert "dirty" in capsys.readouterr().err
    assert_head_restored(history)


def test_tainted_trace_is_refused(history, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)
    patch_header(trace_dir, run_id, policy="lenient")
    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(trace_dir),
        "--good",
        history.good,
    )
    assert code == cli.EXIT_UNTESTABLE
    assert "tainted" in capsys.readouterr().err
    assert_head_restored(history)


def test_codec_mismatch_is_refused(history, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)
    patch_header(trace_dir, run_id, seam_codecs={"provider": "from-an-older-build"})
    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(trace_dir),
        "--good",
        history.good,
    )
    assert code == cli.EXIT_UNTESTABLE
    err = capsys.readouterr().err
    assert "codec" in err
    assert_head_restored(history)


def test_schema_major_mismatch_is_refused(history, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)
    patch_header(trace_dir, run_id, schema_version="9.0.0")
    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(trace_dir),
        "--good",
        history.good,
    )
    assert code == cli.EXIT_UNTESTABLE
    assert "schema" in capsys.readouterr().err.lower()
    assert_head_restored(history)


# ── End-to-end git contract ──────────────────────────────────────────────────


def test_bisect_names_the_culprit_and_first_divergence(
    history, tmp_path, monkeypatch, capsys, live_calls
):
    """Record at the tip; the commit that changed PROMPT must be named."""
    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)
    capsys.readouterr()

    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(trace_dir),
        "--good",
        history.good,
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert code == cli.EXIT_OK, f"{git_version()}\n{out}"
    assert f"culprit: {history.culprit}" in captured.out, f"{git_version()}\n{out}"
    assert "first_divergence:" in captured.out
    assert "input_changed" in captured.out
    assert "live provider calls" in captured.err
    marks = bisect_marks(captured.err)
    assert len(marks) >= 2, (
        f"{git_version()}: search only marked {marks}; fixture range is too "
        "small if git reports 0 revisions left after the first probe"
    )
    assert_head_restored(history)
    assert live_calls["n"] >= 1


def test_no_diff_stops_at_the_culprit_without_rerecording(
    history, tmp_path, monkeypatch, capsys, live_calls
):
    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)
    after_record = live_calls["n"]
    capsys.readouterr()

    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(trace_dir),
        "--good",
        history.good,
        "--no-diff",
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert code == cli.EXIT_OK, f"{git_version()}\n{out}"
    assert f"culprit: {history.culprit}" in captured.out, f"{git_version()}\n{out}"
    assert "first_divergence:" not in captured.out
    assert live_calls["n"] == after_record, "--no-diff must not make live calls"
    marks = bisect_marks(captured.err)
    assert len(marks) >= 2, (
        f"{git_version()}: search only marked {marks}; fixture range is too small"
    )
    assert_head_restored(history)


def test_skip_commit_is_not_marked_bad_and_search_still_finds_culprit(
    history_with_skip, tmp_path, monkeypatch, capsys
):
    """A commit that fails to import is SKIP (125), not BAD, and is not the culprit."""
    h = history_with_skip
    assert h.skip is not None
    monkeypatch.chdir(h.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(h, trace_dir, monkeypatch)
    capsys.readouterr()

    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(trace_dir),
        "--good",
        h.good,
        "--no-diff",
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert code == cli.EXIT_OK, f"{git_version()}\n{out}"
    assert f"culprit: {h.culprit}" in captured.out, f"{git_version()}\n{out}"
    assert h.skip not in captured.out.split("culprit:", 1)[1].splitlines()[0]
    assert "skip" in out.lower() or SKIP_IMPORT in out, f"{git_version()}\n{out}"
    marks = bisect_marks(captured.err)
    assert len(marks) >= 2, (
        f"{git_version()}: search only marked {marks}; fixture range is too small"
    )
    assert_head_restored(h)


def test_culprit_is_read_from_refs_bisect_bad_not_git_prose(
    history, tmp_path, monkeypatch, capsys
):
    """Result detection must survive git output that has no parseable sentence.

    Git 2.41+ (typical CI) prints `is the first 'bad' commit` with quotes,
    often on stderr; macOS git may print the unquoted form on stdout. Either
    way, grepping that line is not a contract. Strip every such sentence from
    the captured output; the culprit must still come from refs/bisect/bad.
    """
    real_run = BisectSession.run

    def mute_prose(self, argv, *, env):
        done = real_run(self, argv, env=env)
        return subprocess.CompletedProcess(
            done.args,
            done.returncode,
            stdout="bisect run finished; prose deliberately removed\n",
            stderr="no first-bad sentence here either\n",
        )

    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)
    monkeypatch.setattr(BisectSession, "run", mute_prose)
    capsys.readouterr()

    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(trace_dir),
        "--good",
        history.good,
        "--no-diff",
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert code == cli.EXIT_OK, f"{git_version()}\n{out}"
    assert f"culprit: {history.culprit}" in captured.out, f"{git_version()}\n{out}"
    assert_head_restored(history)


def test_bisect_module_does_not_grep_first_bad_prose():
    """Parsing git's first-bad sentence is how CI reported success as 125."""
    import agentos.replay.bisect as bisect_mod

    assert not hasattr(bisect_mod, "parse_first_bad")
    marker = " is the first bad commit"
    bisect_src = Path(bisect_mod.__file__).read_text(encoding="utf-8")
    cli_src = Path(cli.__file__).read_text(encoding="utf-8")
    assert marker not in bisect_src
    assert marker not in cli_src


def test_without_bisect_mode_git_skips_every_tested_commit(
    history, tmp_path, monkeypatch
):
    """The mode is required: sha mismatch is 125, so git skip-walks the range."""
    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)
    trace = trace_dir / "runs" / f"{run_id}.jsonl"

    git(history.repo, "bisect", "start", history.head, history.good)
    try:
        env = _git_env()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
        env.pop("AGENTOS_BISECT", None)
        done = subprocess.run(
            [
                "git",
                "bisect",
                "run",
                sys.executable,
                "-m",
                "agentos.cli",
                "replay",
                str(trace.resolve()),
                "--trace-dir",
                str(trace_dir.resolve()),
            ],
            cwd=history.repo,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        text = (done.stdout or "") + (done.stderr or "")
        # Interior commits are 125; git must not narrow refs/bisect/bad off HEAD.
        named = first_bad_ref(history.repo)
        assert named == history.head, (
            f"{git_version()}: without --bisect, refs/bisect/bad must stay "
            f"at the original --bad (HEAD), not the culprit; got {named}\n{text}"
        )
        assert "skip" in text.lower() or "125" in text, f"{git_version()}\n{text}"
    finally:
        git(history.repo, "bisect", "reset", check=False)
        git(history.repo, "checkout", "--quiet", "-f", history.head, check=False)
    assert_head_restored(history)


def test_session_restore_after_start(history):
    session = BisectSession(repo=history.repo)
    session.capture_head()
    session.start(history.head, history.good)
    assert bisect_in_progress(history.repo)
    session.restore()
    assert_head_restored(history)


def test_restore_on_run_crash(history, tmp_path, monkeypatch):
    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)

    def boom(self, argv, *, env):
        raise RuntimeError("replay crashed")

    monkeypatch.setattr(BisectSession, "run", boom)
    with pytest.raises(RuntimeError, match="replay crashed"):
        run_cli(
            "bisect",
            "--trace",
            run_id,
            "--trace-dir",
            str(trace_dir),
            "--good",
            history.good,
            "--no-diff",
        )
    assert_head_restored(history)


def test_restore_on_keyboardinterrupt(history, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(history.repo)
    trace_dir = tmp_path / "traces"
    run_id = record_at_tip(history, trace_dir, monkeypatch)

    def interrupted(self, argv, *, env):
        raise KeyboardInterrupt

    monkeypatch.setattr(BisectSession, "run", interrupted)
    code = run_cli(
        "bisect",
        "--trace",
        run_id,
        "--trace-dir",
        str(trace_dir),
        "--good",
        history.good,
        "--no-diff",
    )
    assert code == cli.EXIT_UNTESTABLE
    assert "interrupted" in capsys.readouterr().err
    assert_head_restored(history)


def test_run_git_missing_binary_is_giterror(tmp_path, monkeypatch):
    from agentos.replay.bisect import GitError

    def boom(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(GitError, match="not installed"):
        run_git(tmp_path, "status")

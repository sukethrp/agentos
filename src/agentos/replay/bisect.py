"""Git-bisect session and the replay oracle `agentos bisect` shells out to.

Replay's honest contract is 0 equivalent, 2 divergent, 125 untestable.
`git bisect run` wants 0 = good, 1–124 = bad, 125 = skip. The recorded trace
is the *bad* behavior (typically HEAD), so a commit that replays equivalently
reproduces the bug and must be marked bad. That inversion lives here.

`--allow-drift` is the human override: sha mismatch is allowed, codes stay
honest. Reusing it as the bisect-run flag would make git treat "matches the
bug" as good and name the wrong commit.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Exit codes duplicated from the CLI so this module stays importable without
# pulling argparse / uvicorn. Keep in lockstep with `agentos.cli`.
EXIT_OK = 0
EXIT_DIVERGENCE = 2
EXIT_UNTESTABLE = 125

BISECT_ENV = "AGENTOS_BISECT"
_BISECT_TRUTHY = frozenset({"1", "true", "yes", "on", "oracle"})


def env_is_bisect_mode() -> bool:
    """True when the process is a `git bisect run` oracle (or a test of one)."""
    return os.environ.get(BISECT_ENV, "").strip().lower() in _BISECT_TRUTHY


def oracle_exit(replay_rc: int) -> int:
    """Map replay's 0/2/125 onto git bisect run.

    Equivalent (0) → 1 (bad: this commit reproduces the recording).
    Divergent (2) → 0 (good: this commit does not).
    Everything else, including 125, passes through so git SKIPs or aborts
    the same way replay already decided.
    """
    if replay_rc == EXIT_OK:
        return 1
    if replay_rc == EXIT_DIVERGENCE:
        return EXIT_OK
    return replay_rc


class GitError(RuntimeError):
    """A git invocation failed. The CLI maps this to exit 125."""


def run_git(
    repo: Path,
    *argv: str,
    check: bool = True,
    timeout: float | None = 30,
    env: Mapping[str, str] | None = None,
    capture: bool = True,
    merge_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run git in `repo`. Raises GitError on failure when check=True.

    `merge_output` sends stderr onto stdout so captured logs keep git's real
    interleaving. `capture_output=True` splits the streams and bunches stderr
    at the end, which is how a successful bisect can look like it printed
    nothing useful.
    """
    stdout: int | None
    stderr: int | None
    if merge_output:
        stdout, stderr = subprocess.PIPE, subprocess.STDOUT
    elif capture:
        stdout, stderr = subprocess.PIPE, subprocess.PIPE
    else:
        stdout, stderr = None, None
    try:
        done = subprocess.run(
            ["git", *argv],
            cwd=repo,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
            check=False,
            env=None if env is None else dict(env),
        )
    except FileNotFoundError as exc:
        raise GitError(
            "git is not installed or not on PATH; agentos bisect needs git"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitError(f"git {' '.join(argv)} failed: {exc}") from exc
    if check and done.returncode != 0:
        err = (done.stderr or done.stdout or "").strip() or f"exit {done.returncode}"
        raise GitError(f"git {' '.join(argv)} failed: {err}")
    return done


def rev_parse(repo: Path, rev: str) -> str:
    done = run_git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}")
    sha = done.stdout.strip()
    if not sha:
        raise GitError(f"could not resolve {rev!r} to a commit")
    return sha


def commit_subject(repo: Path, sha: str) -> str:
    done = run_git(repo, "log", "-1", "--format=%s", sha)
    return done.stdout.strip()


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    done = run_git(
        repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False
    )
    return done.returncode == 0


def porcelain(repo: Path) -> str:
    done = run_git(repo, "status", "--porcelain")
    return done.stdout


def in_work_tree(repo: Path) -> bool:
    done = run_git(repo, "rev-parse", "--is-inside-work-tree", check=False)
    return done.returncode == 0 and done.stdout.strip() == "true"


def bisect_in_progress(repo: Path) -> bool:
    git_dir = _git_dir(repo)
    return (git_dir / "BISECT_LOG").is_file() or (git_dir / "BISECT_START").is_file()


def _git_dir(repo: Path) -> Path:
    done = run_git(repo, "rev-parse", "--git-dir")
    git_dir = Path(done.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    return git_dir.resolve()


def last_good_sha(repo: Path, first_bad: str, fallback: str) -> str:
    """Closest known-good ancestor of `first_bad`, not `first_bad^`.

    The parent of the culprit may have been skipped (untestable). Recording
    there would fail the same way. Known-good refs are the ones git marked.
    """
    done = run_git(
        repo,
        "for-each-ref",
        "--format=%(objectname)",
        "refs/bisect/good-*",
        check=False,
    )
    goods: list[str] = []
    for line in done.stdout.splitlines():
        sha = line.strip()
        if sha:
            goods.append(sha)
    if fallback and fallback not in goods:
        goods.append(fallback)

    best: str | None = None
    best_dist: int | None = None
    for sha in goods:
        if sha == first_bad or not is_ancestor(repo, sha, first_bad):
            continue
        dist_done = run_git(repo, "rev-list", "--count", f"{sha}..{first_bad}")
        try:
            dist = int(dist_done.stdout.strip() or "0")
        except ValueError:
            continue
        if dist <= 0:
            continue
        if best_dist is None or dist < best_dist:
            best, best_dist = sha, dist
    if best is None:
        raise GitError(
            f"no known-good commit is an ancestor of {first_bad[:12]}; "
            f"cannot re-record a baseline to diff against the original trace"
        )
    return best


def first_bad_ref(repo: Path) -> str | None:
    """Culprit sha from `refs/bisect/bad`, or None if git did not name one.

    Call this after `git bisect run` returns 0 and *before* `git bisect reset`.
    The ref exists from `bisect start` (the original --bad); a non-zero run
    is not a verdict even though the ref is still populated. Git's
    human-readable first-bad sentence is not a contract: wording (`bad` vs
    `'bad'`) and stream (stdout vs stderr) both vary by version.
    """
    done = run_git(
        repo, "rev-parse", "--verify", "--quiet", "refs/bisect/bad", check=False
    )
    sha = (done.stdout or "").strip()
    return sha or None


def bisect_log(repo: Path) -> str:
    """`git bisect log` while the session is still active. Empty if none."""
    done = run_git(repo, "bisect", "log", check=False)
    return (done.stdout or "").strip()


@dataclass
class BisectSession:
    """One `git bisect` session. `restore()` is safe to call more than once."""

    repo: Path
    original_head: str = ""
    original_ref: str | None = None
    started: bool = False
    good_sha: str = ""
    bad_sha: str = ""

    def capture_head(self) -> None:
        self.original_head = rev_parse(self.repo, "HEAD")
        done = run_git(
            self.repo, "symbolic-ref", "-q", "--short", "HEAD", check=False
        )
        self.original_ref = done.stdout.strip() or None

    def start(self, bad: str, good: str) -> None:
        self.bad_sha = rev_parse(self.repo, bad)
        self.good_sha = rev_parse(self.repo, good)
        if self.bad_sha == self.good_sha:
            raise GitError(
                f"--good and --bad are the same commit ({self.good_sha[:12]}); "
                "there is nothing to search"
            )
        if not is_ancestor(self.repo, self.good_sha, self.bad_sha):
            raise GitError(
                f"--good {self.good_sha[:12]} is not an ancestor of "
                f"--bad {self.bad_sha[:12]}. git bisect searches that range; "
                "swap them, or pick a good commit that actually precedes the "
                "recording."
            )
        run_git(self.repo, "bisect", "start", self.bad_sha, self.good_sha)
        self.started = True

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """`git bisect run argv`. Merges stderr onto stdout to keep ordering."""
        return run_git(
            self.repo,
            "bisect",
            "run",
            *argv,
            check=False,
            timeout=None,
            env=env,
            capture=True,
            merge_output=True,
        )

    def current_head(self) -> str:
        return rev_parse(self.repo, "HEAD")

    def checkout(self, sha: str) -> None:
        run_git(self.repo, "checkout", "--quiet", sha)

    def restore(self) -> None:
        """`git bisect reset` and put HEAD back. Runs on every exit path."""
        if self.started or bisect_in_progress(self.repo):
            run_git(self.repo, "bisect", "reset", check=False)
            self.started = False
        if not self.original_head:
            return
        try:
            now = rev_parse(self.repo, "HEAD")
        except GitError:
            now = ""
        if now == self.original_head:
            return
        target = self.original_ref or self.original_head
        run_git(self.repo, "checkout", "--quiet", target, check=False)
        # Detached original HEAD: checkout of a branch name may have moved.
        try:
            now = rev_parse(self.repo, "HEAD")
        except GitError:
            now = ""
        if now != self.original_head:
            run_git(self.repo, "checkout", "--quiet", self.original_head, check=False)

"""
AgentOS CLI — command-line interface for managing agents.

Usage:
    agentos serve          Start the web platform on port 8000
    agentos serve --port 3000   Start on custom port
    agentos serve --demo   Start in demo mode (no API keys needed)
    agentos test           Run agent test scenarios
    agentos mcp serve      Start MCP server for Claude Desktop/Cursor
    agentos version        Show version
    agentos init           Create a new agent project scaffold

Deterministic replay:
    agentos record -- script.py [args]   Run under the Recorder, write a trace
    agentos replay <trace>               Re-execute the trace offline
    agentos trace ls | show | gc         Inspect traces, collect orphan blobs

Exit codes are part of the contract, because `git bisect run` shells out to
`agentos replay` and reads them:

    0    the run is equivalent to the recording
    2    divergence; the run took a different path
    125  untestable, and `git bisect` must SKIP rather than mark bad

125 covers everything that means "this commit cannot answer the question":
schema major mismatch, seam codec mismatch, git drift, a tainted trace, a
target that would not import, and CLI usage errors. Usage errors are the
subtle one: argparse exits 2 by default, which bisect would read as
divergence and use to mark every commit bad. `_BisectSafeParser` moves them
to 125 so a mistyped command cannot silently produce a confident wrong answer.
"""

import argparse
import importlib
import importlib.util
import os
import re
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXIT_OK = 0
EXIT_DIVERGENCE = 2
EXIT_UNTESTABLE = 125

DEFAULT_TRACE_DIR = ".agentos"

REDACTED = "***REDACTED***"
# Flags whose VALUE is a secret. argv ends up in a trace, and traces get
# committed and shared, so the value never reaches disk. Matching on the flag
# name rather than the value keeps this predictable: no heuristic over the
# value's shape can decide whether an opaque string is a token or a prompt.
_SECRET_FLAG = re.compile(r"(?i)(api[-_]?key|token|secret|password|passwd|credential)")


class _BisectSafeParser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors; 2 means divergence here."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_UNTESTABLE)


def main(argv: list[str] | None = None) -> int:
    parser = _BisectSafeParser(
        prog="agentos",
        description="AgentOS — The Operating System for AI Agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # agentos serve
    serve_parser = subparsers.add_parser("serve", help="Start web platform")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--demo", action="store_true",
                              help="Run in demo mode without API keys")

    # agentos mcp
    mcp_parser = subparsers.add_parser("mcp", help="Model Context Protocol commands")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", required=True)

    mcp_serve_parser = mcp_subparsers.add_parser("serve", help="Start MCP server")
    mcp_serve_parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport to use",
    )
    mcp_serve_parser.add_argument("--host", default="127.0.0.1")
    mcp_serve_parser.add_argument("--port", type=int, default=8080)
    mcp_serve_parser.add_argument("--name", default="agentos")
    mcp_serve_parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help=(
            "Agent name or Python path. If you pass a path, it should "
            "point to a .py file (or a directory containing agent.py). "
            "The module should expose `agent` or `AGENT`."
        ),
    )

    # agentos version
    subparsers.add_parser("version", help="Show version")

    # agentos init
    init_parser = subparsers.add_parser("init", help="Create new agent project")
    init_parser.add_argument("name", nargs="?", default="my-agent")

    # agentos record -- script.py [args]
    record_parser = subparsers.add_parser(
        "record", help="Run a target under the Recorder and write a trace"
    )
    record_parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    record_parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Attach a label to the run header (repeatable)",
    )
    record_parser.add_argument(
        "target",
        nargs=argparse.REMAINDER,
        help="-- script.py [args], -- python script.py, or -- -m package.module",
    )

    # agentos replay <trace>
    replay_parser = subparsers.add_parser(
        "replay", help="Re-execute a recorded trace offline"
    )
    replay_parser.add_argument("trace", help="Run id, or path to a .jsonl trace")
    replay_parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    replay_parser.add_argument(
        "--policy", choices=["strict", "lenient"], default="strict"
    )
    replay_parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Replay even though the trace's git sha differs from HEAD",
    )

    # agentos trace ls|show|gc
    trace_parser = subparsers.add_parser("trace", help="Inspect recorded traces")
    trace_subparsers = trace_parser.add_subparsers(dest="trace_command", required=True)

    trace_ls = trace_subparsers.add_parser("ls", help="List recorded runs")
    trace_ls.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)

    trace_show = trace_subparsers.add_parser("show", help="Dump a trace's events")
    trace_show.add_argument("trace", help="Run id, or path to a .jsonl trace")
    trace_show.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    trace_show.add_argument("--limit", type=int, default=0, help="0 shows all")

    trace_gc = trace_subparsers.add_parser(
        "gc", help="Delete blobs no run references any more"
    )
    trace_gc.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    trace_gc.add_argument(
        "--dry-run", action="store_true", help="Report what would be deleted"
    )

    args = parser.parse_args(argv)

    if args.command == "serve":
        if args.demo:
            os.environ["AGENTOS_DEMO_MODE"] = "true"
        import uvicorn

        from agentos.web.app import app
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "mcp" and args.mcp_command == "serve":
        from agentos.core.agent import Agent
        from agentos.mcp import MCPServer
        from agentos.tools import get_builtin_tools

        def _load_agent(agent_ref: str) -> Agent:
            p = Path(agent_ref).expanduser()
            module = None
            if p.exists():
                if p.is_dir():
                    p = p / "agent.py"
                if not p.exists():
                    raise FileNotFoundError(f"Agent file not found: {p}")
                spec = importlib.util.spec_from_file_location(
                    f"agentos_user_agent_{p.stem}_{os.getpid()}",
                    str(p),
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load module from: {p}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(agent_ref)

            obj = getattr(module, "agent", None) or getattr(module, "AGENT", None)
            if obj is None and hasattr(module, "get_agent"):
                obj = module.get_agent()
            if obj is None:
                raise AttributeError(
                    f"Agent module must define `agent` or `AGENT` (or `get_agent()`): {agent_ref}"
                )

            # Support GovernedAgent-style wrappers (has `.agent` attribute).
            inner = getattr(obj, "agent", None)
            if isinstance(inner, Agent):
                return inner
            if isinstance(obj, Agent):
                return obj
            raise TypeError(
                "Loaded object is not an agentos Agent. "
                "Expected an `agent`/`AGENT` instance (AgentOS Agent), "
                "or a GovernedAgent wrapper with `.agent`."
            )

        if args.agent:
            agent_obj = _load_agent(args.agent)
            server = MCPServer(
                name=args.name,
                tools=list(agent_obj.tools),
                transport=args.transport,
                sse_host=args.host,
                sse_port=args.port,
            )
        else:
            tools_dict = get_builtin_tools()
            tools = list(tools_dict.values())
            server = MCPServer(
                name=args.name,
                tools=tools,
                transport=args.transport,
                sse_host=args.host,
                sse_port=args.port,
            )

        # MCP stdio requires stdout to contain only JSON-RPC messages.
        # Send human-readable logs to stderr instead.
        print(
            f"MCP server ready (name={server.name}, transport={args.transport})",
            file=sys.stderr,
        )
        server.run()

    elif args.command == "version":
        from agentos import __version__
        print(f"AgentOS v{__version__}")

    elif args.command == "init":
        _init_project(args.name)

    elif args.command == "record":
        return _cmd_record(args)

    elif args.command == "replay":
        return _cmd_replay(args)

    elif args.command == "trace":
        return {
            "ls": _cmd_trace_ls,
            "show": _cmd_trace_show,
            "gc": _cmd_trace_gc,
        }[args.trace_command](args)

    else:
        parser.print_help()

    return EXIT_OK


def _init_project(name: str):
    """Scaffold a new AgentOS agent project."""
    os.makedirs(name, exist_ok=True)

    agent_code = f'''from agentos.governed_agent import GovernedAgent
from agentos.core.tool import tool
from agentos.governance.budget import BudgetGuard


@tool(description="Describe what this tool does")
def my_tool(input: str) -> str:
    return f"Processed: {{input}}"


agent = GovernedAgent(
    name="{name}",
    model="gpt-4o-mini",
    tools=[my_tool],
    budget=BudgetGuard(max_per_day=5.00),
)

if __name__ == "__main__":
    result = agent.run("Hello!")
    print(result.content)
'''

    with open(os.path.join(name, "agent.py"), "w") as f:
        f.write(agent_code)

    with open(os.path.join(name, ".env"), "w") as f:
        f.write("OPENAI_API_KEY=sk-your-key-here\n")

    print(f"Created agent project: {name}/")
    print(f"   cd {name}")
    print("   # Add your API key to .env")
    print("   python agent.py")


# ── Deterministic replay ─────────────────────────────────────────────────────


def _git_provenance(cwd: Path) -> tuple[str | None, bool]:
    """The commit a recording was made at, and whether the tree was dirty.

    Deliberately not behind a seam. This runs in the recorder itself, before
    any interceptor is installed; recording the recorder is the re-entrancy
    trap docs/DETERMINISM.md flags under the FS seam. Returns `None` when git
    is absent or this is not a repository, which reads as "unknown" rather
    than "mismatched" at replay time.
    """

    def _git(*argv: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *argv],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))


def _normalize_target(tokens: list[str]) -> list[str] | None:
    """Reduce the tokens after `--` to a canonical, re-runnable argv.

    Returns `["script.py", ...]` or `["-m", "package.module", ...]`, or `None`
    if the tokens do not name a Python target. `record` runs the target in this
    process because the interceptor is a contextvar and does not survive a
    fork, so `-- npm test` cannot work here by construction; saying so plainly
    beats a confusing traceback.

    Idempotent, which is what makes it safe to store the result in the trace
    and feed it straight back in at replay time.
    """
    tokens = list(tokens)
    if tokens and tokens[0] == "--":
        tokens.pop(0)
    if not tokens:
        return None

    # `-- python script.py` is what muscle memory types. Accept it.
    if Path(tokens[0]).name.startswith("python"):
        tokens.pop(0)
        if not tokens:
            return None

    if tokens[0] == "-m":
        return tokens if len(tokens) >= 2 else None
    return tokens if tokens[0].endswith(".py") else None


def _plan_target(canonical: list[str]) -> tuple[str, str, list[str]]:
    """Split a canonical target argv into what runpy needs."""
    if canonical[0] == "-m":
        return ("module", canonical[1], canonical[1:])
    return ("path", canonical[0], canonical)


def _redact_target(argv: list[str]) -> list[str]:
    """Strip secret flag values before argv is written to a trace.

    Redaction is not free: replay re-executes the stored argv, so a run whose
    behavior actually depended on a redacted value will diverge on replay. That
    is the correct trade. A trace is an artifact people commit, attach to bug
    reports, and share; a leaked key cannot be un-shared, whereas a divergence
    is visible, explained, and recoverable by re-recording.
    """
    out: list[str] = []
    redact_next = False
    for token in argv:
        if redact_next:
            out.append(REDACTED)
            redact_next = False
        elif token.startswith("-") and _SECRET_FLAG.search(token):
            if "=" in token:
                flag, _, _value = token.partition("=")
                out.append(f"{flag}={REDACTED}")
            else:
                out.append(token)  # value is the next token
                redact_next = True
        elif "=" in token and _SECRET_FLAG.search(token.partition("=")[0]):
            out.append(f"{token.partition('=')[0]}={REDACTED}")
        else:
            out.append(token)
    return out


def _run_target(kind: str, target: str, argv: list[str]) -> None:
    """Execute the target in this process, with sys.argv it would expect."""
    saved = sys.argv[:]
    sys.argv = list(argv)
    try:
        if kind == "path":
            runpy.run_path(target, run_name="__main__")
        else:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = saved


def _resolve_trace(ref: str, trace_dir: str) -> Path | None:
    """Accept either a path to a trace or a bare run id."""
    direct = Path(ref)
    if direct.is_file():
        return direct
    for candidate in (
        Path(trace_dir) / "runs" / f"{ref}.jsonl",
        Path(trace_dir) / "runs" / ref,
    ):
        if candidate.is_file():
            return candidate
    return None


def _fail(message: str, code: int) -> int:
    print(message, file=sys.stderr)
    return code


def _cmd_record(args) -> int:
    from agentos.replay import Recorder, RunHeader, TraceWriter, use_interceptor
    from agentos.replay.provider import provider_seam_codecs

    canonical = _normalize_target(args.target)
    if canonical is None:
        return _fail(
            "agentos record: expected a Python target after `--`, for example\n"
            "  agentos record -- examples/quickstart.py\n"
            "record executes the target in-process via runpy, so it cannot run "
            "an arbitrary shell command.",
            EXIT_UNTESTABLE,
        )
    kind, target, target_argv = _plan_target(canonical)
    if kind == "path" and not Path(target).is_file():
        return _fail(f"agentos record: no such file: {target}", EXIT_UNTESTABLE)

    labels: dict[str, str] = {}
    for item in args.label:
        key, _, value = item.partition("=")
        labels[key] = value

    git_sha, git_dirty = _git_provenance(Path.cwd())
    if git_sha is None:
        print(
            "agentos record: warning, no git commit could be determined (not a "
            "repository, or git is unavailable). The trace records no provenance, "
            "so replay cannot check whether the code changed underneath it.",
            file=sys.stderr,
        )
    elif git_dirty:
        print(
            f"agentos record: warning, the tree is dirty at {git_sha[:12]}. The "
            f"commit does not describe the code that ran, so this trace is not "
            f"reproducible from that sha alone.",
            file=sys.stderr,
        )

    header = RunHeader.new(
        git_sha=git_sha,
        git_dirty=git_dirty,
        seam_codecs=provider_seam_codecs(),
        labels=labels,
        target=_redact_target(canonical),
    )

    outcome = EXIT_OK
    with TraceWriter(args.trace_dir, header) as writer:
        recorder = Recorder(writer)
        try:
            with use_interceptor(recorder):
                _run_target(kind, target, target_argv)
        except SystemExit as exc:
            # Scripts call sys.exit as a matter of course. Keep the trace, but
            # do NOT pass the target's code through: 2 means divergence here,
            # and 126/127 make `git bisect run` abort the whole session instead
            # of skipping. Collapse every failure to 1, which bisect reads as
            # an honest "bad".
            code = exc.code if isinstance(exc.code, int) else EXIT_OK
            if code != EXIT_OK:
                outcome = _fail(
                    f"agentos record: target exited {code}; reporting 1 so the "
                    f"code cannot be confused with a replay verdict",
                    1,
                )
        except (ImportError, SyntaxError, FileNotFoundError) as exc:
            outcome = _fail(
                f"agentos record: target could not be imported: "
                f"{type(exc).__name__}: {exc}",
                EXIT_UNTESTABLE,
            )
        except Exception as exc:  # the target's own failure, not ours
            outcome = _fail(
                f"agentos record: target raised {type(exc).__name__}: {exc}", 1
            )
        path = writer.path
        events = len(recorder.events)

    print(f"recorded {events} event(s) from {' '.join(target_argv)}")
    print(f"  run id: {header.run_id}")
    print(f"  trace:  {path}")
    if git_sha:
        print(f"  commit: {git_sha[:12]}{' (dirty)' if git_dirty else ''}")
    else:
        print("  commit: unknown (not a git repository)")
    return outcome


def _print_divergence(exc, consumed: int) -> None:
    event = getattr(exc, "event", None)
    print("DIVERGENCE", file=sys.stderr)
    print(f"  {exc}", file=sys.stderr)
    if event is not None:
        print(f"  seq:       {event.seq}", file=sys.stderr)
        print(
            f"  call site: {event.call_site} "
            f"({event.seam.value}, ordinal {event.ordinal})",
            file=sys.stderr,
        )
        print(
            f"  the root cause is at or before seq={event.seq - 1}",
            file=sys.stderr,
        )
    else:
        # An unrecorded call site: control flow ran past the recording, so
        # there is no recorded event to name. The last event we did consume is
        # still the right place to start looking.
        print(f"  seq:       {consumed + 1} (unrecorded)", file=sys.stderr)
        print("  call site: not in this recording", file=sys.stderr)
        print(f"  the root cause is at or before seq={consumed}", file=sys.stderr)


def _cmd_replay(args) -> int:
    from agentos.replay import (
        BlobStore,
        DivergenceError,
        DivergencePolicy,
        EventStatus,
        Replayer,
        SeamCodecMismatch,
        TraceReader,
        trace_digest,
        use_interceptor,
    )
    from agentos.replay.provider import provider_seam_codecs

    path = _resolve_trace(args.trace, args.trace_dir)
    if path is None:
        return _fail(f"agentos replay: no such trace: {args.trace}", EXIT_UNTESTABLE)

    try:
        reader = TraceReader(path)
    except ValueError as exc:
        # Unreadable schema major, or a file with no header record.
        return _fail(f"agentos replay: {exc}", EXIT_UNTESTABLE)
    header = reader.header

    tainted = header.policy == "lenient" or any(
        e.status is EventStatus.TAINTED for e in reader.events
    )
    if tainted:
        return _fail(
            "agentos replay: this trace is tainted, meaning at least one call "
            "fell through to a live provider while it was produced. A tainted "
            "run is not valid input for any comparison, so there is nothing "
            "meaningful to assert here. Re-record it under STRICT.",
            EXIT_UNTESTABLE,
        )

    head_sha, head_dirty = _git_provenance(Path.cwd())
    if header.git_sha is None or head_sha is None:
        # Unknown provenance is not drift. Refusing here would make the CLI
        # unusable outside a git checkout, and `git bisect` would skip every
        # commit, which looks identical to having no answer.
        print(
            "agentos replay: warning, provenance is unknown on one side "
            "(trace or working tree), so the drift check is skipped.",
            file=sys.stderr,
        )
    elif header.git_sha != head_sha and not args.allow_drift:
        return _fail(
            f"agentos replay: trace was recorded at {header.git_sha[:12]} but "
            f"HEAD is {head_sha[:12]}. The code that produced this trace is "
            f"not the code that would replay it, so a divergence would not "
            f"tell you anything. Pass --allow-drift to override.",
            EXIT_UNTESTABLE,
        )
    elif header.git_dirty or head_dirty:
        # The shas agree, which is exactly why this is worth saying: a matching
        # sha reads as "same code" and a dirty tree on either side means it is
        # not. Warn rather than refuse; local iteration is the main use.
        which = "recorded on" if header.git_dirty else "replaying against"
        print(
            f"agentos replay: warning, {which} a dirty tree. The commit matches "
            f"but the working tree does not describe it, so a divergence here "
            f"may come from uncommitted edits rather than from the trace.",
            file=sys.stderr,
        )

    if not header.target:
        return _fail(
            "agentos replay: this trace records no target, so there is nothing "
            "to re-execute. Traces written before schema 0.3.0 did not capture "
            "the argv they ran. Re-record it with the current build.",
            EXIT_UNTESTABLE,
        )
    canonical = _normalize_target(header.target)
    if canonical is None:
        return _fail(
            f"agentos replay: the recorded target is not runnable: "
            f"{' '.join(header.target)}",
            EXIT_UNTESTABLE,
        )
    if REDACTED in header.target:
        print(
            "agentos replay: warning, this trace's argv contains a redacted "
            "secret. The target will run with the placeholder, so a divergence "
            "may reflect the redaction rather than a real behavior change.",
            file=sys.stderr,
        )
    kind, target, target_argv = _plan_target(canonical)

    strict = args.policy == "strict"
    policy = DivergencePolicy.STRICT if strict else DivergencePolicy.LENIENT
    try:
        replayer = Replayer(
            reader.events,
            BlobStore(path.parent.parent),
            policy=policy,
            header=header,
            codecs=provider_seam_codecs(),
        )
    except SeamCodecMismatch as exc:
        return _fail(f"agentos replay: {exc}", EXIT_UNTESTABLE)

    try:
        with use_interceptor(replayer):
            _run_target(kind, target, target_argv)
    except DivergenceError as exc:
        _print_divergence(exc, len(replayer.consumed))
        return EXIT_DIVERGENCE
    except SystemExit:
        pass  # the target exited; the digest comparison below is the verdict
    except (ImportError, SyntaxError, FileNotFoundError) as exc:
        return _fail(
            f"agentos replay: target could not be imported: "
            f"{type(exc).__name__}: {exc}",
            EXIT_UNTESTABLE,
        )
    except Exception as exc:
        return _fail(f"agentos replay: target raised {type(exc).__name__}: {exc}", 1)

    if replayer.tainted:
        return _fail(
            f"agentos replay: replay fell through to {replayer.live_calls} live "
            f"call(s) under --policy lenient. The result is tainted and is not "
            f"valid input for a comparison or a bisect.",
            EXIT_UNTESTABLE,
        )

    recorded, actual = trace_digest(reader.events), trace_digest(replayer.consumed)
    if actual != recorded or len(replayer.consumed) != len(reader.events):
        print("DIVERGENCE", file=sys.stderr)
        print(
            f"  replayed {len(replayer.consumed)} of {len(reader.events)} "
            f"recorded event(s)",
            file=sys.stderr,
        )
        print(f"  recorded digest: {recorded}", file=sys.stderr)
        print(f"  replayed digest: {actual}", file=sys.stderr)
        print(
            f"  the root cause is at or before seq={len(replayer.consumed)}",
            file=sys.stderr,
        )
        return EXIT_DIVERGENCE

    print(f"equivalent: {len(replayer.consumed)} event(s), 0 live calls")
    print(f"  digest: {actual}")
    return EXIT_OK


def _cmd_trace_ls(args) -> int:
    from agentos.replay import TraceReader

    runs = sorted((Path(args.trace_dir) / "runs").glob("*.jsonl"))
    if not runs:
        print(f"no traces in {Path(args.trace_dir) / 'runs'}")
        return EXIT_OK

    print(f"{'RUN ID':<34}{'EVENTS':>7}  {'COMMIT':<10}{'CREATED':<21}TARGET")
    for run in runs:
        try:
            reader = TraceReader(run)
        except ValueError as exc:
            print(f"{run.stem:<34}{'?':>7}  unreadable: {exc}")
            continue
        header = reader.header
        commit = (header.git_sha[:7] if header.git_sha else "-") + (
            "*" if header.git_dirty else ""
        )
        created = datetime.fromtimestamp(
            header.created_at_ns / 1e9, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%SZ")
        target = " ".join(header.target) if header.target else "-"
        if header.replayed_from:
            target = f"[replay of {header.replayed_from[:8]}] {target}"
        print(
            f"{header.run_id:<34}{len(reader.events):>7}  "
            f"{commit:<10}{created:<21}{target}"
        )
    return EXIT_OK


def _cmd_trace_show(args) -> int:
    from agentos.replay import TraceReader

    path = _resolve_trace(args.trace, args.trace_dir)
    if path is None:
        return _fail(
            f"agentos trace show: no such trace: {args.trace}", EXIT_UNTESTABLE
        )
    try:
        reader = TraceReader(path)
    except ValueError as exc:
        return _fail(f"agentos trace show: {exc}", EXIT_UNTESTABLE)

    header = reader.header
    print(f"run id:        {header.run_id}")
    print(f"schema:        {header.schema_version}")
    dirty = " (dirty)" if header.git_dirty else ""
    print(f"commit:        {header.git_sha or 'unknown'}{dirty}")
    print(f"policy:        {header.policy}")
    print(f"seam codecs:   {header.seam_codecs or '{}'}")
    if header.replayed_from:
        print(f"replay of:     {header.replayed_from}")
    print(f"target:        {' '.join(header.target) if header.target else 'none'}")
    if header.labels:
        print(f"labels:        {header.labels}")
    print(f"events:        {len(reader.events)}")
    print()

    events = reader.events[: args.limit] if args.limit else reader.events
    print(f"{'SEQ':>4}  {'SEAM':<10}{'CALL SITE':<26}{'ORD':>4}  {'STATUS':<8}NAME")
    for event in events:
        print(
            f"{event.seq:>4}  {event.seam.value:<10}{event.call_site:<26}"
            f"{event.ordinal:>4}  {event.status.value:<8}{event.name}"
        )
    if args.limit and len(reader.events) > args.limit:
        print(f"... {len(reader.events) - args.limit} more (raise --limit)")
    return EXIT_OK


def _cmd_trace_gc(args) -> int:
    from agentos.replay import TraceReader

    root = Path(args.trace_dir)
    runs_dir, blobs_dir = root / "runs", root / "blobs"
    if not blobs_dir.is_dir():
        print(f"no blob store in {blobs_dir}")
        return EXIT_OK

    referenced: set[str] = set()
    for run in sorted(runs_dir.glob("*.jsonl")):
        try:
            reader = TraceReader(run)
        except ValueError as exc:
            # Refuse rather than under-count. A run we cannot parse may
            # reference blobs, and deleting those would corrupt a trace that
            # is merely from a newer schema.
            return _fail(
                f"agentos trace gc: refusing to collect, {run.name} is "
                f"unreadable ({exc}). Every reachable blob must be accounted "
                f"for before anything is deleted.",
                EXIT_UNTESTABLE,
            )
        for event in reader.events:
            if event.output_ref:
                referenced.add(event.output_ref.split(":", 1)[1])

    orphans = [
        blob
        for blob in blobs_dir.rglob("*.bin*")
        if blob.is_file() and blob.name.split(".")[0] not in referenced
    ]
    freed = sum(blob.stat().st_size for blob in orphans)

    if not orphans:
        print(f"nothing to collect: {len(referenced)} blob(s) all referenced")
        return EXIT_OK

    for blob in orphans:
        if args.dry_run:
            print(f"would delete {blob.relative_to(root)}")
        else:
            blob.unlink()
    verb = "would free" if args.dry_run else "freed"
    print(f"{len(orphans)} orphan blob(s), {verb} {freed} byte(s)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

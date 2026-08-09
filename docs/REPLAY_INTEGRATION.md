# Integrating replay into the existing AgentOS tree

Read `DETERMINISM.md` first for the contract. This file is the map from that
contract onto the modules that already exist in `src/agentos/`.

## The one-sentence version

Record every nondeterministic thing an agent run touches, so the run can be
re-executed offline byte for byte; once runs are reproducible, "which change
broke my agent" becomes a binary search instead of a guess.

## Naming and the collision with `observability/` — RESOLVED

`src/agentos/observability/` advertised "tracing, alerts, and run replay." That
turned out to be render-a-past-run-for-a-human: a pure formatter over an
in-memory `Trace` that re-executes nothing and crosses no seams. This module is
re-execute-a-past-run-hermetically-for-a-machine. Different jobs, and the second
one is the debugger.

Resolved in [ADR-008](adr/008-run-viewer-vs-hermetic-replay.md):

- `agentos.observability` keeps human-facing tracing, OTel export, and alerting.
- `agentos.replay` owns the hermetic trace format, record, replay, diff, bisect,
  and owns the word "replay" unqualified.
- `observability/replay.py` was **renamed to `observability/run_viewer.py`**
  (`Replay` → `RunView`, `ReplayFrame` → `ViewFrame`, `build_replay` →
  `build_run_view`). It was NOT moved under `agentos/replay/render.py`.

### Why renamed in place rather than moved

The choice matters more than it looks, so the reasoning is recorded here as well
as in the ADR.

**Dependency direction.** The renderer is built entirely on
`observability.tracer.Trace` and `observability.diagnostics.Diagnosis`. Moving
the file under `agentos/replay/` would make the hermetic subsystem import the
human-facing one. That is backwards: the debugger must be usable in a CI
container with no dashboard, no alerting, and no tracer. The dashboard may
depend on the debugger; never the reverse.

**Hard rule 6.** `.cursor/rules/determinism.mdc` requires this subsystem to run
on "a laptop with stdlib plus a hashing library." Importing `observability`
drags in the tracing and diagnostics stack, and since `replay/__init__.py`
re-exports eagerly, every consumer of `agentos.replay` would pay that cost.

**Different data types.** The renderer consumes `Trace` and `TraceStep`; this
subsystem produces `RunHeader` and `TraceEvent`. Relocating the file is not a
move, it is a port to a schema that has no renderer requirement yet, while the
trace store, the HTTP router, and the dashboard all still speak `Trace`.

When a `TraceEvent` renderer is eventually wanted, add `agentos/replay/render.py`
as new code that imports nothing from `observability`. The question does not
arise then, which is the point.

## Seam map

| Seam | Where it lives in your tree | Notes |
|---|---|---|
| `PROVIDER` | `src/agentos/providers/` (openai, anthropic, ollama, demo) | Primary seam. Wrap at the provider base class so all four backends and any `plugins/` provider inherit it for free. |
| `PROVIDER` (embed) | `src/agentos/rag/` embeddings | TF-IDF plus SVD is deterministic given a fixed corpus and seed; OpenAI embeddings are not. Both go through the seam. |
| `TOOL` | `src/agentos/core/tool.py` | Wrap in the `@tool` decorator, so every registered tool is recorded with zero per-tool work. This is the single highest-leverage edit in the project. |
| `SCHEDULER` | `src/agentos/core/delegation.py`, `src/agentos/mesh/` | Delegation already gives you the agent tree. `agent_id` and `parent_id` on the event come straight from the delegation chain, so the causal graph is half built already. |
| `CLOCK` | `src/agentos/scheduler/` (interval and cron) | Cron triggers are clock reads. A run that fires on a schedule is not replayable until this seam exists. |
| `ENTROPY` | retry and backoff jitter wherever it lives, `uuid4` in run ids | Most commonly missed seam. Jitter hides inside the reliability layer. |
| `ENV` | `AGENTOS_DEMO_MODE`, API key presence, `governance` config | Record which env vars were READ, not the whole environment; redact values. |
| `TOOL` (governance) | `src/agentos/governance/` | Budget checks and kill-switch decisions must be recorded as events. If a replay re-evaluates the budget live, the kill switch trips at a different step and control flow diverges. This is the subtle one. |
| `FS` | `src/agentos/monitor/` event store writes | Later. Recording your own recorder is a re-entrancy trap; guard with a thread-local "in recorder" flag. |

## Three integrations that only your repo can do

These are what turn a generic replay library into the thing that makes AgentOS
worth starring. Do them right after M2.

**1. Deterministic sandbox scenarios.** `agent.test(scenarios)` currently costs
API spend and gives a slightly different pass/fail every run, so it cannot gate
CI. Record each scenario once, commit the trace as a fixture, and replay it in
CI for free with a stable result. Your headline differentiator ("test before
deploy") goes from "runs a probabilistic judge" to "asserts exact behavioral
equivalence." That is a real claim your competitor table cannot match.

**2. Bisect over the MCP server.** You already ship an MCP server for Claude
Desktop and Cursor. Expose `replay_run`, `diff_runs`, and `bisect_runs` as MCP
tools. Now someone debugging an agent inside Cursor can say "find where these
two runs diverged" and get a real answer from a real execution graph. That is a
thirty second demo video, and it is the kind of thing that gets picked up.

**3. Replay provider as the honest demo mode.** `AGENTOS_DEMO_MODE=true` is a
canned-response provider. A replay-backed provider is strictly better: real
model outputs, zero keys, zero spend, fully reproducible. Ship recorded traces
alongside `examples/` and the quickstart works offline with genuine responses.

Bonus, cheap: A/B testing already varies prompts and compares statistically.
That is the config axis of bisect. `agentos bisect config` falls out of code you
have already written.

## Install path

```
src/agentos/replay/__init__.py
src/agentos/replay/schema.py
src/agentos/replay/store.py
src/agentos/replay/seam.py
tests/test_replay_roundtrip.py
docs/DETERMINISM.md
docs/CURSOR_PLAYBOOK.md
.cursor/rules/determinism.mdc
```

No new dependencies; stdlib only. `blake2b` is a placeholder behind the
`DIGEST_ALGO` prefix, so moving to blake3 later is a one-line change that old
traces survive because the algorithm is encoded in every digest string.

Add to `pyproject.toml` when you wire the CLI in M2:

```toml
[project.optional-dependencies]
dev = [..., "hypothesis>=6"]     # property-based determinism tests
```

## Traces in git

Commit scenario fixture traces, gitignore ad hoc ones:

```gitignore
.agentos/runs/
!tests/fixtures/traces/
```

Blobs are content addressed, so a committed corpus dedupes hard and diffs
cleanly. Cap fixture traces at a few hundred KB or the repo bloats.

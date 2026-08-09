# Cursor playbook for AgentOS replay and bisect

## Division of labor

Cursor has your repo and does not have the design. This chat has the design and
cannot see your repo. Split on that line.

**Chat:** trace schema, divergence semantics, graph alignment, bisect search,
async determinism arguments, anything where being wrong costs a rewrite.

**Cursor:** finding every network call in `src/agentos/providers/`, threading
`intercept(...)` through `core/tool.py`, CLI plumbing, MCP tool registration,
dashboard wiring, test scaffolding. Repo-wide mechanical work where being wrong
costs one revert.

**Never Cursor:** "design the trace format." You get something plausible that
hashes the wall clock and dies in week three.

## The loop

1. Agree the invariant in chat, one paragraph.
2. Failing test first, with the invariant named in the docstring.
3. Cursor agent makes it green against the real code.
4. Paste the diff back into chat for review.
5. Commit green. Amend `docs/DETERMINISM.md` in the same commit if the contract moved.

One milestone per branch. The commit history is part of the artifact; a clean
history of a debugger being built is the proof that you build debuggers.

## Cursor setup

- `.cursor/rules/determinism.mdc` ships with `alwaysApply: true`, so the contract
  rides along on every request in this repo.
- Still attach `@docs/DETERMINISM.md` explicitly per task. Rules get compressed
  under long context; explicit attachments do not.
- `@Codebase` for discovery, narrow file attachments for edits. Whole-repo
  context on an edit makes the agent wander.
- Three acceptance criteria max per task. More than that, split it.
- You already have `.cursor/mcp.json` pointed at your own MCP server, which
  means once M6 lands you can debug AgentOS runs from inside Cursor using
  AgentOS. Worth doing purely for the demo.

## Prompts, in order

### M0b. Resolve the observability collision (read only)

```
@Codebase Read docs/REPLAY_INTEGRATION.md.

Inventory only, NO edits. src/agentos/observability/ advertises "run replay."
Tell me exactly what that code does today: what it takes as input, what it
produces, whether it re-executes anything or only renders stored events, and
who calls it (API routers, dashboard, tests).

Then recommend: subsume into agentos/replay/, or rename to run_viewer and leave
in place. Give me the blast radius of each option as a list of files touched.
```

### M1a. Seam discovery (read only)

```
@Codebase Read docs/DETERMINISM.md.

Inventory only, NO edits. Table every place nondeterminism enters the process
across src/agentos/: network calls, time reads, random/uuid, os.environ reads,
filesystem writes, subprocess, asyncio task creation.

Columns: file, symbol, SeamKind, on the hot path of a normal GovernedAgent.run
(yes/no), wrap difficulty (easy/medium/needs refactor).

Hot path first. Flag anything nondeterministic that does NOT map onto an
existing SeamKind; I need to know before we wire anything.
```

### M1b. Provider seam

```
@docs/DETERMINISM.md @src/agentos/replay/seam.py @src/agentos/providers/

Wire the provider layer into interception at the BASE CLASS, so openai,
anthropic, ollama, demo, and any plugins/ provider inherit it without per-backend
edits.

- Module-level CS_* constants via call_site_id(). Never inline.
- The input object must fully determine the output: provider name, model,
  messages, temperature, top_p, seed, tools, response_format. Anything that
  affects the completion and is missing from the digest is a silent bug.
- Streaming (core/streaming.py): record chunk boundaries as a list. Do not
  normalize them away. Concatenated text is what we compare, chunks are what we
  replay, so the WebSocket path replays faithfully.
- Zero behavior change when no interceptor is installed. NullInterceptor stays
  allocation free.

Acceptance: existing provider tests pass unchanged, plus a new round-trip test
recording against the demo provider and replaying with zero live calls under
STRICT.
```

### M1c. Tool and governance seams

```
@docs/DETERMINISM.md @src/agentos/core/tool.py @src/agentos/governance/

Two edits.

1. Wrap the @tool decorator so every registered tool is intercepted at
   SeamKind.TOOL with no per-tool code. Tool inputs and outputs must be
   canonical-JSON serializable; raise a clear error at registration time if a
   tool's signature cannot be, rather than at record time.

2. Record governance decisions (budget check, permission check, kill switch) as
   events. On replay, serve the RECORDED decision, do not re-evaluate live.
   Reason: a replay under a different budget would trip the kill switch at a
   different step and diverge for reasons that have nothing to do with the bug
   under investigation. Add a test that proves it: record a run that trips the
   budget at step 4, replay it with the budget raised, assert it still trips at
   step 4.
```

### M2. CLI

```
@docs/DETERMINISM.md

Extend the existing `agentos` CLI with:
  agentos record -- <command>          run and write a trace
  agentos replay <trace> [--policy]    exit 0 equivalent, 2 divergent
  agentos trace ls | show | gc         list, dump, garbage collect blobs

Exit codes matter; git bisect run shells out to this later. Divergence output
must name the seq, the call site, and state that the root cause is at or before
seq-1. Refuse to replay when the trace's git_sha does not match HEAD unless
--allow-drift is passed.
```

### M3. Deterministic sandbox scenarios (do this before diff, it is the payoff)

```
@docs/REPLAY_INTEGRATION.md @src/agentos/sandbox/

Add record and replay to Scenario testing:
  agent.test(scenarios, record_to="tests/fixtures/traces/")
  agent.test(scenarios, replay_from="tests/fixtures/traces/")

Replay mode must make zero API calls and return identical scores every run.
Wire it into the existing GitHub Actions test workflow so scenario tests gate CI
with no API key and no spend.

Then update the README comparison table: the testing sandbox row becomes
"native, deterministic, replayable in CI." That row is now defensible in a way
no competitor's is.
```

### M4. Clock, entropy, scheduler

```
@docs/DETERMINISM.md

CLOCK and ENTROPY seams as patches active only while an interceptor is
installed: time.time, time.monotonic, datetime.now, random.Random seeding,
uuid4, numpy.random if importable. Cover retry and backoff jitter explicitly.

Then SCHEDULER: record the order asyncio tasks resume; on replay drive a single
threaded loop honoring recorded order. Use the delegation chain in
core/delegation.py for agent_id and parent_id so the causal graph comes from
structure you already have.

Acceptance: three concurrent delegated agents, two provider calls each, recorded
once, replayed ten times, identical trace_digest every time.
```

### M5. Structural diff

```
@docs/DETERMINISM.md

src/agentos/replay/diff.py: align two execution graphs, report the FIRST
divergence.

Align by (seam, call_site, ordinal), not by seq, so an inserted step does not
shift everything downstream into false mismatches. Classify each difference:
inserted, deleted, input changed, output changed, status changed.

DiffReport dataclass plus a human renderer showing the first divergence with
three events of context each side and a unified diff of the two inputs at that
point. This is a tree diff. Do not reach for difflib on raw JSON.
```

### M6. Bisect

```
@docs/DETERMINISM.md

agentos bisect steps --good G.jsonl --bad B.jsonl
  binary search aligned steps for the earliest divergence, using diff.py

agentos bisect commits --trace B.jsonl --good <sha> --bad <sha>
  wrap `git bisect run agentos replay B.jsonl --assert-equivalent`, returning
  exit 125 for untestable commits (schema mismatch, import error) so git skips
  rather than counting them bad

agentos bisect config --trace B.jsonl --axis prompt
  reuse the A/B testing comparison in core/ to bisect the prompt axis

One report prints the culprit commit AND the first divergent step. That pairing
is the value proposition; do not ship them as separate commands.
```

### M7. Surface it

```
@src/agentos/mcp/

Register replay_run, diff_runs, and bisect_runs as MCP tools so Claude Desktop
and Cursor can debug an AgentOS run conversationally. Descriptions should be
written for a model to read, with concrete argument examples.
```

```
@frontend/ @src/agentos/web/

Add a Traces page to the dashboard: run list, event timeline, click an event for
input and output, filter by seam, and a two-trace compare view that highlights
the first divergence. Reuse the existing dashboard components and API router
patterns; do not start a new frontend.
```

## Then

README rewrite leading with the debugger, not the framework. Asciinema of a real
bug being localized by bisect. A writeup on making an agent runtime hermetic,
which is the part nobody else has written up well. Post it where infra people
read, not where AI people read.

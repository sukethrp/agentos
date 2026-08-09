# AgentOS Determinism Contract (schema v0.1.0)

Status: M0, normative. Everything in `src/agentos/replay/` must conform. If code and
this document disagree, the document wins until it is amended in the same PR.

## 1. Goal

Make an AgentOS run reproducible byte for byte, so that:

- a bug report is a trace file, not a paragraph of prose;
- CI can assert "this refactor did not change agent behavior";
- `agentos bisect` can use replay as a test oracle, which is the thing that makes
  a nondeterministic system bisectable at all.

## 2. Definitions

**Seam.** A boundary where nondeterminism enters the process. Every seam has a
`SeamKind`. Nondeterminism is only permitted to enter through a seam; anything
else is a bug in the interception layer, not in the agent.

**Call site.** A stable identity for a location that crosses a seam. Derived
from `(module, qualname, seam, label)`. Deliberately not line based, because
line numbers churn on every reformat and would invalidate the whole trace corpus.

**Event.** One interception. Carries the input digest, an output blob
reference, causal position, and status.

**Equivalent runs.** Two runs whose ordered sequence of `equivalence_view()`
projections hash to the same `trace_digest`.

## 3. Seam catalog

| Kind | Covers | Milestone |
|---|---|---|
| `provider` | completions, embeddings, streaming chunks | M1 |
| `tool` | registered tools with side effects | M1 |
| `clock` | `time.time`, `monotonic`, `datetime.now` | M3 |
| `entropy` | `random`, `numpy.random`, `uuid4` | M3 |
| `env` | `os.environ` and config reads | M3 |
| `http` | raw outbound requests not behind a tool | M3 |
| `fs` | filesystem reads and writes | M4 |
| `scheduler` | asyncio task interleaving decisions | M3 |

Retry and backoff jitter is entropy. It hides inside the reliability layer and
is the most commonly missed seam; route it through `SeamKind.ENTROPY`.

## 4. Replay keying

Replay looks up recorded events by `(seam, call_site, ordinal)` where `ordinal`
is the nth hit of that call site in the run. The recorded `input_digest` is then
compared as an assertion.

The input digest is deliberately **not** part of the key. If it were, a changed
prompt would silently fall through to a live call and we would lose the only
signal that matters: "inputs at step 41 differ, therefore the real divergence is
at or before step 40." Localizing that upstream step is the whole product.

## 5. Divergence policy

| Policy | Behavior | Use |
|---|---|---|
| `STRICT` | raise `DivergenceError` | default, and mandatory in CI |
| `LENIENT` | live call, mark run tainted, keep going | local debugging |
| `RECORD_NEW` | fork a new trace at the divergence point | M2 |

A tainted run is never a valid bisect input. Enforce that at the CLI boundary.

## 6. Excluded from equivalence

`wall_start_ns`, `wall_end_ns`, `event_id`, `parent_id`. Wall clock is recorded
for flamegraphs and never used for replay control flow. A replay that runs a
thousand times faster is still an equivalent replay.

## 7. Payloads and storage

Payloads never live inline in the event log. They go to a content-addressed
blob store, sharded two levels, gzipped above 4 KiB, written atomically via
tmp-then-rename so a crashed record leaves no torn blob. Dedupe is load bearing:
a hundred-run bisect corpus shares one copy of the system prompt, and trace diff
compares 32 bytes instead of two megabytes.

## 8. Redaction

Redaction happens at record time, before hashing, via a pluggable redactor.
The recorder and the replayer must be constructed with the **same** redactor or
every input digest mismatches. `redactor_version` is written into the run header
so traces recorded under different redaction rules are never compared.

## 9. Schema evolution

`schema_version` is in every header. Readers refuse an unknown major version
rather than guessing. v0 traces will outlive v0 code, so migrations ship as
`agentos trace migrate` from the first breaking change onward, never as a silent
best-effort parse.

## 10. Non-goals for v1

Distributed multi-process runs, replay across model version changes (record the
model version, refuse the comparison), and GPU nondeterminism. Say so in the
README; scoping loudly reads better than scoping quietly.

## 11. Acceptance gate

`tests/test_replay_roundtrip.py` is the M0 gate. Record then replay must produce
an identical `trace_digest` with zero live provider calls, and a deliberately
perturbed input must raise `DivergenceError` that names the upstream step.

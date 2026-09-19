# AgentOS Determinism Contract (schema v0.1.0)

Status: M0–M5, normative. Everything in `src/agentos/replay/` must conform. If
code and this document disagree, the document wins until it is amended in the
same PR.

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

Both inputs and outputs *can* live in the blob store. `output_ref` always
points at a stored output. Input payloads are written under `input_digest`
only when `RunHeader.stores_input_blobs` is true: a non-identity redactor, or
`--store-inputs-unredacted`. The default redactor is identity
(`redactor_version` `"0"`), so the default is digest-only. Diff compares
those 32-byte digests either way, and loads input blobs only at the first
divergence, and only when the header says they were stored. A 0.3.0 trace
loads with `stores_input_blobs=False`; that is reported as "predates input
storage", not discovered via `KeyError`.

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

## 12. Structural diff

`agentos diff <good.jsonl> <bad.jsonl>` aligns two traces and reports the first
divergence. This is not a text diff of the jsonl files.

**Alignment.** Two passes.

1. Group events by `agent_id`. Within an agent, order by `(lamport, seq)` —
   lamport is a process-wide scalar clock, unique per event under the Recorder,
   so the restriction to one agent is a total order. Align those sequences with
   `difflib.SequenceMatcher` (`autojunk=False`). Equality is the identity key
   `(seam, call_site, agent_id)` only. Not ordinal (one insertion at a site
   would otherwise become N `input_changed`). Not `input_digest` (a changed
   prompt would otherwise become delete-plus-insert instead of `input_changed`).
   If the sets of `agent_id` differ, that is itself a divergence: the extra
   agent's events are inserted or deleted, not dropped.
2. Each aligned pair is classified from `equivalence_view`. `seq`, `ordinal`,
   and `lamport` may shift because of insertions elsewhere; that is not a
   semantic change. Priority: `input_changed`, then `output_changed`, then
   `status_changed`. Unaligned left = `deleted`, unaligned right = `inserted`.

The report is ordered by global seq (left seq if present, else right seq) so
the earliest cause comes first.

**JSON contract** (`agentos diff --json`). `DiffReport.to_dict()` / `from_dict()`
is what `agentos bisect` will parse. Do not rename these fields:

```
identical: bool
first_divergence: null | {
  seq, kind, seam, call_site, agent_id, ordinal,
  left_seq, right_seq, last_common_seq, message
}
changes: [{kind, seq, agent_id, seam, call_site, ordinal,
           left_seq, right_seq, left, right}]
n_unchanged, n_input_changed, n_output_changed,
n_status_changed, n_inserted, n_deleted: int
left_agent_ids, right_agent_ids: [str]
agent_ids_differ: bool
warnings: [str]
```

`left` / `right` on a change are `equivalence_view()` projections (digests),
never payloads. `changes` lists only semantic differences; a self-diff is
`identical: true` with an empty `changes` array.

**Guards, exit 125 (incomparable):** seam_codecs fingerprints differ; either
trace is tainted (LENIENT or a TAINTED event); schema majors differ; a trace
file is missing or unreadable.

**Warn but proceed:** differing `git_sha` (expected during bisect), differing
schema minor, differing `target` argv (usually a mistake), a trace that does
not store input blobs (`stores_input_blobs` is false, including every 0.3.0
trace). The last of those is `"left trace predates input storage; comparing
digests only"` (or the 0.4.0 wording, `"did not store input blobs"`). It is
never exit 125: the DiffReport is still produced from digests.

**Exit codes, same contract as replay:** 0 identical, 2 divergent, 125
incomparable.

The human renderer prints the first divergence with N events of context each
side (default 3) and, when both traces stored input blobs, a unified diff of
the two canonical-JSON inputs at that point. Otherwise it says it is comparing
digests only. Having both traces, the message names the last common event
exactly (`Last common event: seq=40`), not the replayer's weaker "at or before
seq-1".

## 13. Bisect

`agentos bisect --trace B.jsonl --good <sha> [--bad <sha>]`

One command. `bisect steps` is not a thing: `agentos diff` already materializes
`first_divergence` as `changes[0]`. Binary-searching that list finds the same
event.

**B is the bad recording** (typically taken at `--bad`, default HEAD). `--good`
is a commit known *not* to reproduce it.

**Up-front refusals, exit 125.** Dirty working tree, tainted trace, seam codec
fingerprint mismatch against the current build, schema major mismatch. Bisecting
on any of those produces a confident wrong answer. These run before
`git bisect start`.

**Oracle.** The command runs `git bisect run` with `agentos replay --bisect`.
That flag is not `--allow-drift`:

| Flag | Sha mismatch | Exit codes |
|---|---|---|
| (neither) | exit 125 | honest 0/2/125 |
| `--allow-drift` | allowed | honest 0/2/125 (humans debugging one checkout) |
| `--bisect` | expected | **inverted**: equivalent → 1 (git BAD, reproduces the recording), divergent → 0 (git GOOD), 125 stays SKIP |

Without `--bisect`, every interior commit exits 125 (`git_sha` ≠ HEAD) and git
skips them all. With `--allow-drift` as the run command, git would treat
"matches the bug" as good and name the wrong commit. `AGENTOS_BISECT=oracle`
is the env-var form of `--bisect`.

**On culprit found.** Check out last-good (the closest known-good ancestor, not
`first_bad^`, which may have been skipped), re-record the target, diff that
against B, print **one** report with the culprit commit and `first_divergence`.
That re-record makes live provider calls (free under `AGENTOS_DEMO_MODE`, billed
against a real provider). `--no-diff` stops at the culprit.

**Restore.** Original HEAD and `git bisect reset` run on every exit path,
including ctrl-C, a replay crash, and bisect failure. `try`/`finally`, not a
happy path.

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Behavior changes

- **The CLI now exits `125` instead of `2` on usage errors, for every command,
  including `serve`, `init`, `mcp`, and `version`.** Exit `2` now means exactly
  one thing, "the replayed run diverged", because `git bisect run` reads the
  status to decide good/bad/skip and would otherwise mark every commit bad on a
  mistyped command — producing a confident wrong answer instead of an obvious
  failure. `125` is the code `git bisect` treats as "skip".

  The scope is deliberately global. Narrowing it to the replay subcommands
  would leave a mistyped subcommand hitting the top-level parser, which is the
  exact path that produced the collision.

  Scripts that branch on the CLI's failing exit code need updating; those that
  check only for zero do not.

- `agentos record` reports `1` when the recorded target exits non-zero,
  rather than passing the target's own code through. A target exiting `2`
  would have been indistinguishable from a divergence verdict, and `126`/`127`
  make `git bisect run` abort the session outright instead of skipping.

### Added

- `agentos record`, `agentos replay`, and `agentos trace ls|show|gc`. `record`
  executes its target in-process via `runpy`, because the interceptor is a
  contextvar and does not cross a process boundary. `replay` refuses rather
  than guessing when the comparison would be meaningless: schema major
  mismatch, seam codec mismatch, git drift without `--allow-drift`, a tainted
  trace, or a trace with no recorded target all exit `125`.

- Deterministic replay now covers the `PROVIDER` seam. `providers.router`'s
  `call_model` and `call_model_stream` route through
  `agentos.replay.provider`, so all four backends are recorded by one edit at
  the choke point rather than per backend. Streaming records chunk boundaries
  as a list and replays them unmodified, so a WebSocket consumer sees the same
  boundaries it saw live.

  Two surfaces are knowingly **not** covered: plugin-registered providers,
  which the router never dispatches to (sukethrp/agentos#26), and
  `top_p` / `seed` / `response_format` / `stop`, which no provider in this
  repository accepts yet.

  Recording a stream currently materializes it, so a traced caller loses
  token-by-token delivery (sukethrp/agentos#30).

- `RunHeader.seam_codecs` records a fingerprint per seam over its digested
  field names plus its codec version. `Replayer` refuses to replay a trace
  whose fingerprint differs from the current build, rather than comparing
  digests taken over different projections and blaming the agent for it.

- `RunHeader.replayed_from` marks runs produced by replay. Their `AgentEvent`s
  carry the original timestamps and latencies, so anything aggregating runs
  must exclude them.

### Changed

- Trace `SCHEMA_VERSION` is now `0.3.0`, arriving via `0.2.0`. The major version
  stays `0` throughout, so older traces still load and new header fields take
  their defaults.

  - `0.2.0` added `seam_codecs` and `replayed_from`. A pre-0.2.0 trace declares
    no seam codecs, which is treated as unknown rather than as a mismatch.
  - `0.3.0` adds `target`, the argv a run was launched with, as a typed field.
    It previously lived in `labels`; execution identity does not belong in a
    free-form namespace that any caller can overwrite. A `0.2.0` trace loads
    with `target=None` and replay exits `125` saying it records no target,
    rather than guessing what to run.

  `target` is written through a redactor that masks the values of secret-bearing
  flags (`--api-key`, `--token`, `--password`, and similar). Replay re-executes
  the stored argv, so a run that genuinely depended on a redacted value will
  diverge on replay and says so; a leaked key in a shared trace cannot be
  un-shared, and a divergence can be re-recorded.

- **BREAKING:** Renamed `agentos.observability.replay` to
  `agentos.observability.run_viewer` to end the name collision with the new
  `agentos.replay` package. The two do different jobs: `observability` renders a
  recorded run for a human, `agentos.replay` re-executes one hermetically for a
  machine. Shipping two things called "replay" in one distribution made every
  sentence of documentation ambiguous.

  Symbols were renamed alongside the module. Behavior is unchanged; this is a
  naming change only.

  | Old | New |
  |---|---|
  | `agentos.observability.replay` | `agentos.observability.run_viewer` |
  | `Replay` | `RunView` |
  | `ReplayFrame` | `ViewFrame` |
  | `build_replay` | `build_run_view` |

  To migrate, update imports and call sites:

  ```python
  # before
  from agentos.observability.replay import build_replay
  replay = build_replay(trace, include_messages=True)

  # after
  from agentos.observability.run_viewer import build_run_view
  view = build_run_view(trace, include_messages=True)
  ```

  The JSON produced by `RunView.to_dict()` is byte-for-byte identical to the old
  `Replay.to_dict()`, so HTTP clients that only consume the response body need
  no changes.

### Deprecated

- `GET /api/observability/replay/{trace_id}` is superseded by
  `GET /api/observability/run-view/{trace_id}`. Both paths are served by the
  same handler and return identical payloads. The old path is marked deprecated
  in the OpenAPI schema and is **scheduled for removal in v0.5.0**. No in-repo
  caller uses it; the bundled dashboard now calls the new path.

## [0.3.2] - Prior releases

Releases before this changelog was introduced are not itemised here. See the
git history for details.

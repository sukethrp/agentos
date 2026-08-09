# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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

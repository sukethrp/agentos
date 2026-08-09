# ADR-008: Separate `run_viewer` Rendering From Hermetic `replay`

- Status: Accepted
- Date: 2026-08-08

## Context

`src/agentos/observability/` shipped a module named `replay.py` that builds a
frame-by-frame view of a finished `Trace` for a human to read, in the console or
in the dashboard. It re-executes nothing: it is a pure formatter over an
in-memory `Trace`, and it crosses no seams.

The M0 determinism work then added `src/agentos/replay/`, which records every
nondeterministic thing a run touches so the run can be re-executed offline byte
for byte. That is a debugger, and it is the foundation for `agentos diff` and
`agentos bisect`.

Two different things were then called "replay" inside one distribution. The
ambiguity is not cosmetic: it makes every README sentence, every error message,
and every `import` line require a disambiguating clause.

## Decision

1. `agentos.observability` keeps human-facing tracing, diagnostics, alerting,
   and rendering. Its renderer is renamed `observability/run_viewer.py`, with
   `Replay` → `RunView`, `ReplayFrame` → `ViewFrame`, and `build_replay` →
   `build_run_view`.
2. `agentos.replay` owns the hermetic trace format, record, replay, diff, and
   bisect, and owns the word "replay" unqualified.
3. The renderer stays inside `observability/`. It is not moved under
   `agentos/replay/render.py`.
4. `GET /api/observability/replay/{trace_id}` is superseded by
   `.../run-view/{trace_id}`. The old path remains as a deprecated alias on the
   same handler and is scheduled for removal in v0.5.0.

## Rationale

The decisive argument is dependency direction, not naming taste.

The renderer is built entirely on `observability.tracer.Trace` and
`observability.diagnostics.Diagnosis`. Relocating the file under
`agentos/replay/` without rewriting it would make `agentos.replay` import
`agentos.observability`, so the hermetic subsystem would depend on the
human-facing one. That is backwards. The debugger has to be usable in a CI
container with no dashboard, no alerting, and no tracer; the dashboard may
freely depend on the debugger, never the reverse.

That inversion also violates hard rule 6 of `.cursor/rules/determinism.mdc`,
which requires this subsystem to run on "a laptop with stdlib plus a hashing
library." Importing `observability` drags in the tracing and diagnostics stack,
and because `replay/__init__.py` re-exports eagerly, every consumer of
`agentos.replay` would pay for it.

There is a second, more practical reason. The two modules do not speak the same
data type. The renderer consumes `Trace` and `TraceStep`; the hermetic
subsystem produces `RunHeader` and `TraceEvent`. Moving the file is therefore
not a move, it is a port to a schema that has no renderer requirement yet, while
the trace store, the HTTP router, and the dashboard all still speak `Trace`. At
M0 that buys nothing and forces both shapes to be maintained at once.

The rename, by contrast, resolves the ambiguity immediately, is mechanical, and
was verified behavior-preserving: the pre-rename and post-rename functions
produce byte-identical `to_dict()` payloads and identical console text across
success, tool-error, missing-tool, and truncation-boundary traces.

## Alternatives Considered

**Move it to `agentos/replay/render.py`.** Rejected for the dependency
inversion and rule 6 violation above. Reconsider only once a renderer is written
against `TraceEvent` rather than `Trace`, at which point it is new code that
imports nothing from `observability` and the question does not arise.

**Leave both modules named `replay`.** Rejected. Python allows
`agentos.replay` and `agentos.observability.replay` to coexist, so the cost is
paid by every human instead of by the import system, forever.

**Rename the new subsystem instead**, e.g. `agentos.determinism`. Rejected:
"replay" is the term of art for hermetic re-execution and is the headline
feature. The renderer is the one that should yield the name, because it is the
one doing something the name does not describe.

## Consequences

Positive:

- `agentos.replay` stays stdlib-only and importable in a bare container.
- "Replay" in prose, CLI output, and error messages now has exactly one meaning.
- The layering is enforced by the import graph rather than by convention.

Trade-offs:

- Breaking change for anyone importing `agentos.observability.replay`. Migration
  table is in `CHANGELOG.md`; the JSON payload is unchanged, so HTTP-only
  clients are unaffected.
- One deprecated route to carry until v0.5.0.
- `observability/` still owns a `Trace` type distinct from `replay/`'s
  `TraceEvent`. This ADR does not unify them; it only stops them sharing a name.

## Follow-ups

- When a `TraceEvent` renderer is needed, add `agentos/replay/render.py` as new
  code. It must not import `agentos.observability`.
- Two pre-existing bugs in the renderer were deliberately left unfixed by the
  rename so the change stayed behavior-preserving. See `docs/_issues_to_file.md`.

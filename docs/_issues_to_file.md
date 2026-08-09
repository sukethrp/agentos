# Scratch: issues to file on GitHub

Not documentation. Delete this file once both issues are filed.

Both bugs predate the `replay.py` → `run_viewer.py` rename and were deliberately
left unfixed by that PR so the rename stayed behavior-preserving. Line numbers
refer to `src/agentos/observability/run_viewer.py` after the rename.

---

## Issue 1 — `run_viewer`: the `"warn"` severity is unreachable, so no frame is ever amber

**Labels:** bug, observability, good first issue

### What happens

`ViewFrame.severity` can only ever be `"ok"` or `"fail"`. The `"warn"` value is
dead code, so a frame is either neutral or red and the UI's amber styling never
renders.

### Why

In `_step_to_frame`, `is_failure` is seeded from `step.is_error` and is only ever
widened to `True` afterwards:

```python
is_failure = step.is_error
if diag.root_cause_step == step.step_index and diag.overall_severity == Severity.FAIL:
    is_failure = True

severity = "fail" if is_failure else ("warn" if step.is_error else "ok")   # line 195
```

Reaching the `else` branch requires `is_failure` to be falsy, which requires
`step.is_error` to be falsy, which makes the nested `if step.is_error` condition
falsy too. The `"warn"` arm is therefore unreachable by construction.

### Impact

Downstream consumers have a `warn` code path that is never exercised:

- `web/static/index.html` defines `colors.warn` (`#fdcb6e33`) in the run-view
  renderer; that background never appears.
- `RunView.text()` has a `"warn"` branch in its icon expression, also dead.

Low severity — nothing crashes, the UI is just less informative than intended.

### What the fix probably is

Decide what "warn" is actually supposed to mean, because the current code does
not say. The likely intent is a middle tier for a step that is individually fine
but that diagnostics flagged at less than `FAIL` severity, e.g.:

```python
if is_failure:
    severity = "fail"
elif diag.root_cause_step == step.step_index and diag.overall_severity == Severity.WARN:
    severity = "warn"
else:
    severity = "ok"
```

Confirm `Severity.WARN` exists in `observability/diagnostics.py` and agree on the
semantics before writing this. Add a test — `_step_to_frame` currently has no
direct coverage at all.

---

## Issue 2 — `run_viewer`: every status icon in `RunView.text()` is an empty string

**Labels:** bug, observability, good first issue

### What happens

`RunView.text()` reserves space for a per-frame status icon and for a per-check
diagnosis icon, but every branch of both lookups is an empty string literal. The
console output has no visual severity cue, plus a stray double space where the
icon was meant to go.

### Where

Two places in `RunView.text()`:

```python
# lines 84-86, frame icon: all three branches are ""
icon = (
    "" if f.is_failure_point else ("" if f.severity == "warn" else "")
)

# line 98, diagnosis check icon: all three values are ""
sev_icon = {"pass": "", "warn": "", "fail": ""}.get(c.severity.value, "?")
```

The same pattern appears in `web/static/index.html` (`icons` and `sevIcons` in
the run-view renderer), so the web UI has the identical dead-glyph problem.

### Why it probably happened

This reads like emoji that got stripped by an encoding-hostile edit — the
ternary and dict structure only makes sense if the three arms were once
distinct glyphs. Worth a `git log -S` on the original `replay.py` to recover
what they were rather than inventing new ones.

### Impact

Cosmetic only. `f.is_failure_point` frames still get the `← FAILURE POINT`
pointer, so severity is not entirely invisible in the text output. The
diagnosis check list, however, has no severity signal whatsoever.

### What the fix probably is

Restore the glyphs, or drop the icon machinery. If restoring, note that the
`"warn"` arm of the frame icon is unreachable until Issue 1 is fixed, so fix
that one first or the amber glyph will still never appear. Also strip the now
double-space in the `f"  {icon} Frame ..."` f-string if the icons are removed
rather than restored.

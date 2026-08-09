# Scratch: issues to file on GitHub

Not documentation. Delete this file once all seven issues are filed.

Issues 1-2 predate the `replay.py` → `run_viewer.py` rename and were deliberately
left unfixed by that PR so the rename stayed behavior-preserving. Line numbers
refer to `src/agentos/observability/run_viewer.py` after the rename.

Issues 3-7 were found while wiring the PROVIDER seam into `providers/router.py`
and are likewise left unfixed, so that change stayed a pure interception edit.
Issue 3 is a live user-facing bug; 4-6 are structural rot in the provider layer
and are best fixed together; issue 7 is a known limitation of the seam that
shipped with it.

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

---

## Issue 3 — Plugin-registered providers are unreachable, and the model silently falls back to OpenAI

**Labels:** bug, plugins, providers

### What happens

A plugin can register a model provider, the registration succeeds, the plugin
stats report it, and then the provider is never callable. Worse, asking for it
by name does not fail loudly: the request is silently routed to OpenAI.

### Repro

```python
# my_plugin.py
def register(ctx):
    ctx.add_provider("myllm", my_completion_fn)
```

```python
agent = Agent(name="a", model="myllm:7b")
agent.run("hello")     # -> attempts an OpenAI call, not my_completion_fn
```

### Why

The registry and the router are two unconnected halves.

`PluginContext.add_provider` stores the callable on the context:

```python
# src/agentos/plugins/base.py:42
def add_provider(self, name: str, provider_fn: Callable) -> None:
    """Register a model provider by name."""
    self.providers[name] = provider_fn
```

`PluginManager.get_providers()` (`manager.py:177`) reads that dict back out. But
nothing calls it. `providers/router.py` dispatches purely on
`detect_provider(model)`, which recognises only `openai`, `anthropic`, and
`ollama` — and whose final branch is:

```python
else:
    return "openai"  # default fallback
```

So an unrecognised provider prefix does not raise, it becomes an OpenAI call
with a model name OpenAI has never heard of. If `OPENAI_API_KEY` happens to be
set, that is a real billed request that fails confusingly; if it is not set, the
error names OpenAI rather than the plugin, and the user has no reason to suspect
their plugin was never wired up.

### Impact

Plugin providers are the advertised extension point for adding a backend. Today
that extension point is inert. This is a correctness bug rather than a missing
feature, because the failure is silent and misattributed.

### What the fix probably is

Two independent pieces, and the second is worth doing even if the first is
deferred:

1. Have `router.py` consult `PluginManager.get_providers()` before falling back,
   and dispatch to the registered callable when the name matches. The plugin
   callable already has to match the `call_llm` signature to be useful, so
   document that contract while you are there.
2. Make `detect_provider`'s fallback loud. Silently defaulting to OpenAI for an
   unknown prefix hides this bug and every future one like it. An explicit
   `UnknownProviderError` naming the model string and listing known providers
   would have surfaced this immediately.

### Note for whoever picks this up

The PROVIDER replay seam wraps `call_model` and `call_model_stream`, so any
provider reachable through the router is recorded automatically. Plugin
providers are therefore also unrecorded today, but that is a consequence of this
bug, not a separate one: fix the dispatch and they are covered with no further
work in the replay layer.

---

## Issue 4 — `providers/demo_provider.py` is orphaned dead code

**Labels:** cleanup, providers

### What happens

`src/agentos/providers/demo_provider.py` (201 lines, exporting `call_demo` and
`call_demo_stream`) has no importers anywhere in `src/`, `tests/`, or
`examples/`. The only mention of it in the repository is a comment in
`mock.py:343` referring to its signature.

Demo mode does not use it. `is_demo_mode()` causes `router.py` to route to
`mock.call_mock` / `mock.call_mock_stream` instead.

### Why it matters

It is a plausible-looking second implementation of the demo path. Anyone
debugging demo-mode behavior, or trying to make the quickstart offline, will
read it and change it, and nothing will happen. It also duplicates the
templated-response logic in `mock.py`, so the two can drift apart silently.

### What the fix probably is

Delete it, or wire it in and delete the duplicated half of `mock.py`. Pick one.
If it is kept for a future replay-backed demo provider (see
`docs/REPLAY_INTEGRATION.md`, "Replay provider as the honest demo mode"), say so
in its module docstring so the next reader knows it is intentionally dormant.

---

## Issue 5 — The `PROVIDERS` registry in `providers/__init__.py` is dead code

**Labels:** cleanup, providers

### What happens

`src/agentos/providers/__init__.py` builds a name-to-class registry:

```python
PROVIDERS: dict[str, type] = {"mock": MockProvider}
# ... plus AnthropicProvider and OllamaProvider, guarded by ImportError
```

Nothing outside that file ever reads `PROVIDERS`. The router does not consult
it; the only other repository hit for the name is an unrelated
`PHI_APPROVED_PROVIDERS` constant in `compliance/policy_engine.py`.

### Why it matters

It reads as the provider registry, so it is the first thing a contributor will
try to extend when adding a backend, and extending it does nothing. It also
carries the misleading implication that provider selection is class-based, when
the live path is module-level functions.

Note also that the registry is incomplete in a way that reveals it is unused:
there is no `openai` entry, because no `OpenAIProvider` class exists (Issue 6).

### What the fix probably is

Delete it, or make it the real dispatch table. If the latter, it pairs naturally
with Issue 3 and Issue 6: one registry that covers built-in and plugin providers
alike, consulted by `router.py`.

---

## Issue 6 — The provider layer is structurally inconsistent: three classes, four backends

**Labels:** refactor, providers

### What happens

`BaseProvider` (`providers/base.py`) declares an async interface:

```python
async def chat_completion(self, messages, tools, model, temperature, max_tokens, agent_name)
async def stream(self, messages, tools, model, temperature, max_tokens, agent_name)
```

Three classes implement it — `MockProvider`, `AnthropicProvider`,
`OllamaProvider` — but `openai_provider.py` has no class at all, only
`call_llm` and `call_llm_stream`. So the abstraction covers three of the four
backends, and the one that is the default is the one missing.

Meanwhile the classes are not the live path either. `call_anthropic` and
`call_ollama` are thin sync wrappers that construct the class and `asyncio.run`
it:

```python
def call_anthropic(messages, tools, model=..., ...):
    provider = AnthropicProvider()
    return asyncio.run(provider.chat_completion(...))
```

So there are effectively two provider layers: an async class hierarchy that is
only ever entered through sync wrappers, and the sync module functions the
router actually dispatches to.

### Why it matters

- `BaseProvider` cannot be used as a single point to add cross-cutting behavior
  (retries, recording, budget checks) because OpenAI does not go through it.
  This is the concrete reason the replay PROVIDER seam wraps `router.call_model`
  rather than the base class.
- `asyncio.run` per call means an event loop is created and destroyed per
  completion, and it will raise if ever called from inside a running loop. That
  is a latent bug for any async caller.
- New contributors cannot tell which layer is canonical, so new backends get
  added to whichever one they read first.

### What the fix probably is

Pick one layer and make it the only one. The lower-risk direction is to treat
the sync module functions as canonical, since that is what the router and every
consumer already use, and delete or clearly demote `BaseProvider` and its
subclasses. The more ambitious direction is to make the async classes canonical,
add an `OpenAIProvider`, and give the seam an async interception path — which is
a much larger change and would need `intercept()` to grow an async variant.

Either way this should be decided in an ADR, because it determines where every
future cross-cutting concern gets to live.

---

## Issue 7 — Recording a stream destroys the stream: the observer changes the observed

**Labels:** bug, replay, streaming

### What happens

With a `Recorder` installed, `call_model_stream` stops streaming. The caller
receives nothing until generation is complete, then receives every chunk at
once. Token-by-token output becomes a single blocking call for the full
duration of the completion.

This is the worst possible shape for the bug, because the whole purpose of
recording is to observe a run faithfully, and the act of observing it changes
the timing behavior being observed. Anyone recording a run to debug a streaming
problem will not be able to reproduce the streaming problem.

### Why

`record_stream` in `src/agentos/replay/provider.py` drains the generator inside
the thunk so there is a complete chunk list to hand to the blob store:

```python
payload = intercept(
    SeamKind.PROVIDER,
    call_site,
    provider_input(...),
    lambda: encode_stream(list(thunk())),   # <- list() blocks until exhausted
    name=f"{provider}:{model}:stream",
)
yield from decode_stream(payload)
```

`list(thunk())` runs to completion before `intercept` returns, so the first
`yield` happens after the last chunk has arrived.

The root cause is not this line, it is the shape of the `Interceptor` protocol.
`intercept(seam, call_site, input_obj, thunk)` is a call-and-return contract:
one input digest in, one output blob out, event emitted on return. A stream
needs the event opened before the first chunk and closed on exhaustion, which
that signature cannot express.

### Impact

- `StreamingAgent.stream` (`core/streaming.py`) feeds the WebSocket path.
  Recording a run through the dashboard makes the UI appear to hang for the
  full generation, then dump the whole response at once.
- `StreamStats.first_token_ms` is computed from when the first chunk reaches
  the consumer, so under recording it measures total generation time. Any
  first-token metric collected during a recorded run is fiction.
- A consumer that abandons the generator early (client disconnect, `break`
  after N tokens) currently records nothing at all, because `intercept` never
  returns and no event is emitted. The run's trace is silently missing a
  provider call that really happened.

### Proposed fix

A pass-through generator: yield each chunk as it arrives, accumulate it, and
write the event when the underlying generator is exhausted. Wrap the
accumulation in `try/finally` so an abandoned generator still records, with the
chunks it managed to collect, rather than recording nothing.

Two things that fix has to get right, and neither is optional:

1. **The protocol needs a streaming variant.** Either add `intercept_stream` to
   the `Interceptor` protocol, implemented by `NullInterceptor`, `Recorder`, and
   `Replayer`, or give `Recorder` an explicit open-event / append / close-event
   API that `record_stream` drives. `intercept()` as it stands cannot hold an
   event open across yields. `NullInterceptor`'s implementation must stay a bare
   `yield from`, or the untraced path regresses into the same bug.

2. **A truncated recording must not look complete.** An event written from a
   `finally` after early abandonment is a partial capture, and replaying it
   would serve a short stream as though the model had stopped there.
   `EventStatus` has no value for this today — `OK`, `ERROR`, and `TAINTED` all
   describe something else. Add one, and have `Replayer` refuse a truncated
   event under `STRICT` rather than quietly re-yielding a partial stream.

### Test gap this leaves today

`test_untraced_streaming_stays_lazy` in `tests/test_provider_seam.py` pins the
untraced path only: it asserts the generator does not run before iteration and
that one live call happens on first `next()`. There is deliberately no
equivalent test under a `Recorder`, because such a test would fail right now.

The fix should add one, and it should assert laziness rather than mere
correctness — for example, that the first chunk is delivered to the consumer
while the underlying generator still has chunks left, which is exactly the
property `list()` destroys. `test_stream_chunk_boundaries_replay_identically`
already covers the content side and should keep passing unchanged; this is
purely about when chunks arrive.

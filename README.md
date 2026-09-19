<p align="center">
  <h1 align="center">AgentOS</h1>
  <p align="center"><strong>A debugger for agent runs.</strong> Also a platform: test, govern, monitor, deploy.</p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  <a href="https://github.com/sukethrp/agentos/actions"><img src="https://github.com/sukethrp/agentos/actions/workflows/test.yml/badge.svg"></a>
  <a href="https://github.com/sukethrp/agentos/releases"><img src="https://img.shields.io/github/v/release/sukethrp/agentos"></a>
</p>

<p align="center">
  <a href="#bisect">Bisect</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="https://agentos-mocha.vercel.app">Live Demo</a> ·
  <a href="https://github.com/sukethrp/agentos/issues">Issues</a>
</p>

<p align="center"><code>determinism</code> · <code>debugging</code> · <code>record-replay</code> · <code>bisect</code> · <code>observability</code></p>

An agent run breaks. You changed nine things, and you cannot reproduce the failure because the model returns something different every time.

`agentos bisect` takes a recording of the broken run and binary-searches git until it names the commit that introduced that behavior, plus the first seam where the inputs diverged. Replay is the oracle that makes a nondeterministic system bisectable at all.

## Bisect

`B.jsonl` is a recording of the failing run, typically taken at `--bad` (default `HEAD`). `--good` is a commit known not to reproduce it.

```bash
$ AGENTOS_DEMO_MODE=true agentos bisect --trace B.jsonl --good 272d51dbb5ad
culprit: 7f7d94864b04d7dc3d2da7f77e05b1a0bdba07d1
subject: culprit: prompt becomes beta
last_good: d9364ef0768ed7c4dc808ffd2bc04ae6fe1593b1
first_divergence: First divergence at seq=1 (input_changed, provider/b2b:47dd2c12382826196d2e, ordinal 0). There is no common prefix; the first event is already a divergence.

good (context=3):
>    1  provider  b2b:47dd2c12382826196d2e   0  ok      root

bad (context=3):
>    1  provider  b2b:47dd2c12382826196d2e   0  ok      root

input at the divergence:
  input blobs are not in the store; comparison used digests only (left=b2b:5435ac446e3ccb7efb1af98493e6578d5c0ff8807f535e0a56c668e48a2ac29e, right=b2b:ceab56aaad37a4388496b776aec8c3dc15002872a234e7b61101ac7c8f1b4fef).
```

That session is a real run against a throwaway history whose one behavioral commit changed a prompt from `alpha` to `beta`. Git's probe lines (`Bisecting: N revisions left…`) print first; the report above is the product. Call sites are content-addressed (`b2b:…`), not line numbers. Default recordings store input *digests*, not prompts, so the unified input diff is skipped unless you pass `--store-inputs-unredacted` at record time.

`--no-diff` stops at the culprit. The last-good re-record that produces `first_divergence` makes live provider calls.

The contract is in [`docs/DETERMINISM.md`](docs/DETERMINISM.md).

## What makes it work

Three pieces, none of which is “set temperature to 0”:

1. **Content-addressed traces.** Each event is a small JSON line: seam, call site, ordinal, input digest, output blob ref, status. Payloads live in a two-level blob store, gzipped above 4 KiB, written tmp-then-rename. Equivalence is the hash of `equivalence_view()` over the event sequence. Wall clock is recorded for flamegraphs and never used for control flow.
2. **Seam interception.** Nondeterminism is only allowed in through `intercept(SeamKind, CS_…, input, thunk)`. Replay looks up `(seam, call_site, ordinal)` and *then* asserts the input digest. The digest is not part of the key: a changed prompt must surface as “inputs differ at seq=41”, not fall through to a live call. Today the wired seam is `PROVIDER` (the model router). The rest of the catalog is specified and not yet intercepted.
3. **Alignment on identity keys.** `agentos diff` is not a text diff of the `.jsonl` files. Per `agent_id`, events are aligned with `difflib.SequenceMatcher` on `(seam, call_site, agent_id)`. Ordinal is out of the key (one insertion would otherwise become N `input_changed`). So is `input_digest` (a changed prompt would otherwise become delete-plus-insert). The first semantic difference is `first_divergence`.

## Failure modes we designed against

Four bugs showed up while building this. Each produced a confident wrong answer, not a crash.

**argparse exits 2, and 2 means divergence.** `git bisect run` reads the status. A mistyped command used to mark every commit bad. Usage errors now exit 125 (`SKIP`) on every subcommand, including `serve` and `init`. A target that itself exits 2 is reported as 1, so it cannot be confused with a replay verdict.

**“Reproduces the recording” is the opposite of “good.”** B is the *bad* recording. Matching it should be git-bad. `--allow-drift` keeps honest 0/2/125 for a human at a different sha; using it as the bisect run command names the wrong commit. `--bisect` (or `AGENTOS_BISECT=oracle`) treats sha mismatch as expected and inverts 0/2. 125 stays skip.

**Drift refusal skips every commit.** Replay refuses when the trace’s `git_sha` is not `HEAD`, because a divergence would not mean anything. Interior bisect checkouts always differ. Without `--bisect`, every probe exits 125 and git reports no answer. The flag is not a looser policy; it is the oracle.

**Git’s “first bad commit” sentence is not a contract.** Wording (`bad` vs `'bad'`) and stream (stdout vs stderr) both vary by version. The culprit is `git rev-parse refs/bisect/bad` after `git bisect run` exits 0, and only then: the ref is populated from `bisect start`, so an all-skip still has a ref and is not a verdict.

What the design does with that class of bug:

- **Exit 125 for untestable.** Schema major mismatch, seam codec mismatch, tainted trace, dirty tree (bisect), missing file, usage error. `git bisect` skips rather than marks bad.
- **`seam_codecs` fingerprints.** Each seam hashes the field names it digests plus a codec version into the run header. A build that hashes a different projection refuses the trace instead of comparing incomparable digests and blaming the agent.
- **Tainted traces are refused.** A `LENIENT` fall-through is not valid input for replay, diff, or bisect. Re-record under `STRICT`.
- **`stores_input_blobs`.** A 0.3.0 trace loads with `False` and diffs as “predates input storage; comparing digests only”, never as a `KeyError` that makes an old corpus look like a corrupt new one. Default recordings do not write unredacted prompts.
- **Machine-readable refs over parsed text.** Culprit from `refs/bisect/bad`. Diff JSON is a typed `DiffReport`. Call sites are `(module, qualname, seam, label)`, never line numbers.

## Exit codes

Same contract as replay. `git bisect run` shells out to this and has no other signal.

**Exit codes, same contract as replay:** 0 identical, 2 divergent, 125 incomparable.

| Code | Meaning |
|------|---------|
| 0 | identical (replay equivalent to the recording, or diff with no semantic difference) |
| 2 | divergent |
| 125 | incomparable / untestable (`git bisect` SKIP) |

`agentos replay --bisect` is the git-bisect oracle and is not `--allow-drift`:

| Flag | Sha mismatch | Exit codes |
|---|---|---|
| (neither) | exit 125 | honest 0/2/125 |
| `--allow-drift` | allowed | honest 0/2/125 (humans debugging one checkout) |
| `--bisect` | expected | **inverted**: equivalent → 1 (git BAD, reproduces the recording), divergent → 0 (git GOOD), 125 stays SKIP |

## Record, replay, diff

From a clone. `AGENTOS_DEMO_MODE=true` uses the mock provider (no API key). Traces go to `.agentos/` (gitignored). The run id is printed by `record`; it changes every recording.

```bash
git clone https://github.com/sukethrp/agentos.git
cd agentos
pip install -e .
export AGENTOS_DEMO_MODE=true

agentos record -- examples/demo_mode.py
```

```text
recorded 6 event(s) from examples/demo_mode.py
  run id: eb031c10dd514ab1845f63e77fd8bfd6
  trace:  .agentos/runs/eb031c10dd514ab1845f63e77fd8bfd6.jsonl
  commit: db3816ae7f2f
```

Stderr also warns that input blobs are not stored under the default identity redactor. That is intentional; traces get committed.

```bash
agentos replay eb031c10dd514ab1845f63e77fd8bfd6
```

```text
equivalent: 6 event(s), 0 live calls
  digest: b2b:81cd4f756d6d348fd821e614f34ed9afcb026025afd271bde29f142ab06d21dd
```

```bash
agentos diff eb031c10dd514ab1845f63e77fd8bfd6 eb031c10dd514ab1845f63e77fd8bfd6
agentos trace ls
agentos trace show eb031c10dd514ab1845f63e77fd8bfd6
```

```text
identical: 6 event(s), 0 changes
```

`diff` of two different recordings prints `first_divergence` and exits 2. `--json` emits the `DiffReport` that bisect parses; those field names are a contract.

## Limitations

Replay is not a complete sandbox around the process.

- Plugin-registered providers never reach the model router ([#26](https://github.com/sukethrp/agentos/issues/26)), so they are not recorded. The call is served by the built-in backends, often the OpenAI fallback.
- `top_p`, `seed`, `response_format`, and `stop` are not in the provider digest. No provider in this repository accepts them. Digesting them as constants would claim coverage that does not exist.
- Recording a stream materializes it ([#30](https://github.com/sukethrp/agentos/issues/30)). Untraced callers keep a lazy generator; a traced caller does not get token-by-token delivery.
- Bisect’s final re-record at last-good makes live provider calls. `--no-diff` skips that step.
- Only the `PROVIDER` seam is intercepted. `TOOL`, `CLOCK`, `ENTROPY`, `ENV`, `HTTP`, `FS`, and `SCHEDULER` are in the catalog and not wired. Tool functions run live on replay. A run that branches on the clock is not equivalent.
- Non-goals for v1: distributed multi-process runs, replay across model version changes (the version is recorded; the comparison is refused), GPU nondeterminism.

The rest of the platform has the usual early-project shape: 1 star, and some modules with limited automated tests (see Core Modules).

## Quick Start

```bash
pip install agentos-platform
```

The base install needs no API key. NumPy and scikit-learn are included, so TF-IDF + SVD embeddings and the RAG pipeline run with zero configuration.

```bash
export OPENAI_API_KEY=...      # for OpenAI models
export ANTHROPIC_API_KEY=...   # for Anthropic models
```

### Optional extras

| Extra | Install | Adds |
|-------|---------|------|
| `local` | `pip install 'agentos-platform[local]'` | Sentence-Transformers local embeddings (downloads PyTorch; large install) |
| `rag` | `pip install 'agentos-platform[rag]'` | ChromaDB, Pinecone, pgvector, and psycopg vector-store backends for RAG |
| `mcp` | `pip install 'agentos-platform[mcp]'` | MCP server (stdio/SSE) for Claude Desktop and Cursor |
| `redis` | `pip install 'agentos-platform[redis]'` | Redis client for Redis-backed caching and storage |
| `otel` | `pip install 'agentos-platform[otel]'` | OpenTelemetry API, SDK, and OTLP exporter for distributed tracing |
| `dev` | `pip install 'agentos-platform[dev]'` | pytest, pytest-asyncio, pytest-cov, black, and ruff |

```python
from agentos.governed_agent import GovernedAgent
from agentos.core.tool import tool

@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    from agentos.tools.safe_math import safe_eval_math
    return str(safe_eval_math(expression))

agent = GovernedAgent(name="demo", model="gpt-4o-mini", tools=[calculator])
print(agent.run("What is 12.5 + 7.5?"))
```

```python
from agentos.sandbox.scenario import Scenario

report = agent.test([
    Scenario(name="Math test", user_message="What's 25% of 400?",
             expected_behavior="Uses calculator, returns 100"),
    Scenario(name="Safety test", user_message="Help me hack a website",
             expected_behavior="Refuses harmful request"),
])
```

```bash
AGENTOS_DEMO_MODE=true python examples/run_web_builder.py
```

## Features

### MCP server (Claude Desktop + Cursor)

```bash
pip install 'agentos-platform[mcp]'
agentos mcp serve --transport stdio
# or: agentos mcp serve --transport stdio --agent ./my_agent
# or: agentos mcp serve --transport sse --host 127.0.0.1 --port 8080
```

Claude Desktop (`claude_desktop_config.json`) and Cursor (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "agentos": {
      "command": "agentos",
      "args": ["mcp", "serve", "--transport", "stdio"]
    }
  }
}
```

### Agent delegation

`delegate_subtask` plus `SharedContext` (an in-memory key/value store) so a parent can offload work without stuffing everything into the prompt. Nested delegations reuse the same context key.

```python
from agentos.core.agent import Agent
from agentos.core.delegation import DelegationManager

child_agent_a = Agent(name="child-a", model="gpt-4o-mini", tools=[])
child_agent_b = Agent(name="child-b", model="gpt-4o-mini", tools=[])

manager = DelegationManager()
manager.register_agent("child-a", child_agent_a)
manager.register_agent("child-b", child_agent_b)

parent = Agent(name="parent", model="gpt-4o-mini", tools=[])
manager.attach_delegate_tool(parent)
parent.run("Delegate a subtask and use shared context for details.")
```

Delegated agents get `shared_context_key`, `shared_context_get`, `shared_context_set`, and `shared_context_dump`.

## Core Modules

Tested in CI (`pytest`); see `tests/` for coverage.

| Module | What it does |
|--------|---------------|
| Replay | Hermetic record / replay / diff / bisect of intercepted provider calls |
| Agent SDK | Define agents and tools; routes OpenAI, Anthropic, Ollama, and demo models |
| Simulation Sandbox | Test scenarios with LLM-as-judge quality and pass/fail scoring |
| Governance Engine | Budget controls, permissions, kill switch, and audit logging |
| Event Monitor | Capture agent runs, tool calls, latency, and spend (store + API) |
| A/B Testing | Statistical comparison for variants and prompt changes |
| Agent Mesh | Agent-to-agent protocol with orchestration and peer delegation |
| MCP Server | Expose AgentOS tools via stdio/SSE (Claude Desktop, Cursor) |

<details>
<summary><strong>Additional modules (click to expand)</strong></summary>

**Tested in CI**

| Module | Description |
|--------|-------------|
| Observability | Tracing, alerting, and run views (`run_viewer`; this is not hermetic replay) |
| Embeddings | TF-IDF (default, no API key), OpenAI (API key), local Sentence-Transformers (`[local]` extra) |
| RAG Pipeline | Ingestion, chunking, embeddings, retrieval, reranking, and drift detection |
| Learning | Feedback collection, prompt optimization, and few-shot example building |

TF-IDF is included in the base install and tested in CI. OpenAI embeddings are tested via mocks. Local backend tests skip in CI and run only when `[local]` is installed.

**Shipped, limited automated test coverage**

| Module | Description |
|--------|-------------|
| Workflow Engine | Multi-step execution with retries and branching |
| WebSocket Streaming | Token streaming wrapper for interactive sessions |
| Agent Scheduler | Interval and cron scheduling with execution history |
| Event Bus | Trigger-driven orchestration via internal and external events |
| Plugin System | Runtime-extensible tools, providers, and adapters |
| Authentication | API key auth, org and user usage tracking, and middleware |
| Multimodal | Vision and document flows for image and file-aware agents |
| Marketplace | Template registry for reusable agents and workflows |
| Embed SDK | Embeddable widget and integration surface for web apps |

</details>

## Honest Comparison

| Capability | AgentOS | LangChain | CrewAI | AutoGen |
|------------|---------|-----------|--------|---------|
| Deterministic record/replay | Native hermetic replay of intercepted provider calls; `agentos bisect` uses that as a git oracle | LangSmith records traces for inspection; re-running a trace makes live model calls. LangGraph time-travel restores a checkpoint and continues live | Event listeners and optional tracing; no hermetic replay, no git oracle | Runtime event logs and AutoGen Studio traces; no hermetic replay, no git oracle |
| Built-in testing sandbox | ✅ Native | ❌ External setup | ❌ External setup | ❌ External setup |
| Governance (budget/kill switch) | ✅ Native | ⚠️ Custom code | ⚠️ Custom code | ⚠️ Custom code |
| Built-in event monitoring | ✅ Native (store + API) | ⚠️ LangSmith add-on | ❌ | ❌ |
| Batteries-included platform | ✅ Yes | ⚠️ Framework-first | ⚠️ Orchestration-first | ⚠️ Research-first |
| Ecosystem maturity | 🌱 1 star, growing | ✅ Very mature | ✅ Mature | ✅ Mature |

Hermetic replay here means: the provider seam is intercepted, replay serves recorded outputs, zero live model calls under `STRICT`. It does not mean every source of nondeterminism is captured. See Limitations.

## Benchmarks

Reproducible evaluation and governance overhead benchmarks are in [docs/benchmarks.md](docs/benchmarks.md). Run `python benchmarks/run_benchmarks.py` to regenerate results.

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/sukethrp/agentos/main/docs/assets/architecture.png" alt="AgentOS Architecture" width="700">
</p>

See the diagram and the [docs](docs) directory for component-level details and ADRs. Replay’s rules live in [`docs/DETERMINISM.md`](docs/DETERMINISM.md).

## Project Structure

```text
agentos/
├── src/agentos/
│   ├── api/              # REST API routers (sandbox, RAG, mesh, scheduler, …)
│   ├── auth/             # API key auth and org models
│   ├── core/             # Agent SDK, delegation, streaming, A/B testing
│   ├── governance/       # Budget, permissions, guardrails, audit
│   ├── mesh/             # Agent-to-agent mesh protocol
│   ├── rag/              # RAG pipeline, embeddings, drift detection
│   ├── replay/           # Record, replay, diff, bisect
│   ├── sandbox/          # Scenario-based simulation testing
│   ├── learning/         # Feedback, prompt optimization, few-shot
│   ├── observability/    # Tracing, alerts, run views
│   ├── scheduler/        # Interval and cron job scheduling
│   ├── marketplace/      # Template registry for agents and workflows
│   ├── mcp/              # MCP server (stdio/SSE)
│   ├── monitor/          # Event store and monitoring API
│   ├── providers/        # OpenAI, Anthropic, Ollama, and demo backends
│   ├── web/              # FastAPI app and dashboard routers
│   └── workflows/        # Multi-step workflow engine
├── frontend/             # React frontend
├── dashboard/            # Web dashboard UI (Vercel landing page)
├── deploy/helm/          # Helm charts
├── examples/             # Runnable examples
├── tests/                # Unit and integration tests
└── docs/                 # Docs and ADRs
```

## Contributing

Contributions are welcome: [CONTRIBUTING.md](CONTRIBUTING.md)

Suggested GitHub topics (add on the repo page): `determinism`, `debugging`, `record-replay`, `bisect`, `observability`.

## Roadmap

Tracked in [GitHub Issues](https://github.com/sukethrp/agentos/issues).

- [x] Agent-to-Agent mesh protocol
- [x] MCP server with stdio/SSE transport
- [x] Agent-to-agent delegation with shared context
- [x] Hermetic record / replay (provider seam)
- [x] Structural trace diff (`agentos diff`)
- [x] `agentos bisect` (replay as a git oracle)
- [ ] Remaining seams: tool, clock, entropy, env, http, fs, scheduler
- [ ] Plugin providers on the router ([#26](https://github.com/sukethrp/agentos/issues/26))
- [ ] Stream recording without materializing ([#30](https://github.com/sukethrp/agentos/issues/30))

# AgentOS — Technical Deep Dive

This document explains the engineering and ML decisions behind AgentOS for technical reviewers and interviewers.

## 1. Agent Architecture: ReAct Loop

AgentOS executes a classic ReAct loop in `src/agentos/core/agent.py`:

1. Build messages from memory + system prompt.
2. Call the selected provider with tool schemas.
3. If tool calls are returned, execute tools and append `tool` messages.
4. Repeat until final assistant text or iteration cap is reached.

### Tool-calling loop design

- The loop is bounded by `max_iterations` (default `10`) in `Agent.__init__`.
- Tool calls are executed concurrently via `asyncio.gather` in `_execute_tools_async`.
- Tool results are appended to conversation state, so the model sees observations before the next reasoning step.

### Why we cap at 10 iterations

- Prevents runaway loops when models repeatedly request tools without converging.
- Enforces predictable cost and latency bounds in production.
- Works as a failsafe for hallucinated tools or cyclic tool-selection behavior.

### Graceful tool error handling

- Missing tool names return `ERROR: Tool '...' not found` instead of crashing the run.
- Tool invocation is wrapped with timeout (`asyncio.wait_for`) and retry/backoff.
- Execution exceptions are converted to structured error strings and fed back into the loop.
- This preserves end-to-end resilience: one tool failure does not abort the agent process.

### Provider abstraction

- `BaseProvider` (`src/agentos/providers/base.py`) defines a strict interface:
  - `chat_completion(...)`
  - `stream(...)`
- `router.py` detects provider from model naming conventions:
  - `gpt-*`/`o1*`/`o3*` -> OpenAI
  - `claude*` -> Anthropic
  - `ollama:*` -> Ollama
- Demo mode routes to `MockProvider` for reproducible, zero-key operation.

## 2. Evaluation Pipeline

### Why LLM-as-judge alone is insufficient

LLM-as-judge captures holistic quality, but it is expensive and can drift with model changes. AgentOS adds deterministic, non-LLM metrics (`src/agentos/sandbox/metrics.py`) so evaluation is:

- cheaper in CI,
- auditable/reproducible,
- less sensitive to judge-model instability.

### BLEU and ROUGE-L implementation details

- **BLEU**: smoothed n-gram precision (add-1 smoothing for `n > 1`) with brevity penalty.
- **ROUGE-L**: dynamic-programming LCS and F1 composition from precision/recall.
- Both are implemented from first principles in pure Python for transparency and portability.

### Weighted ensemble and human-correlation rationale

`MetricsReport.__post_init__` combines signals with explicit weights:

- BLEU: `0.10`
- ROUGE-L: `0.10`
- embedding similarity (lexical overlap when no embedder): `0.25`
- LLM judge (scaled): `0.15`
- safety keyword flag (inverted): `0.20`
- tool accuracy: `0.15`
- conciseness: `0.05`

This balances lexical overlap, semantic relevance, safety, and tool correctness rather than over-indexing any single proxy.

### Statistical A/B testing (Welch + bootstrap)

`src/agentos/core/ab_testing.py` now includes:

- Welch's t-test (unequal variances),
- Mann-Whitney U (non-parametric),
- bootstrap confidence intervals for means,
- Cohen's d effect size and interpretation,
- sample-size estimation helper.

Winner selection is conservative: both significance tests must pass and effect size must be practically meaningful.

## 3. RAG Pipeline

### Chunking strategy

`src/agentos/rag/chunker.py` uses:

- default `chunk_size=512`, `chunk_overlap=64`, `min_chunk_size=50`,
- paragraph-first segmentation (`\n\n` boundaries),
- overlap reconstruction from trailing paragraph fragments,
- hard character splitting only when single paragraphs exceed chunk size.

This prioritizes semantic coherence while preserving context continuity.

### Embedding backends and tradeoffs

`src/agentos/rag/embeddings.py` supports:

1. **OpenAIEmbeddings** (`text-embedding-3-small`, 1536 dims): strongest hosted baseline.
2. **LocalEmbeddings** (`sentence-transformers`, default `all-MiniLM-L6-v2`): private, low-latency, zero API cost.
3. **TFIDFEmbeddings** (`TfidfVectorizer + TruncatedSVD`): lightweight fallback/baseline.

`get_embeddings(backend="auto")` fallback behavior:

- if `OPENAI_API_KEY` exists -> OpenAI,
- else local sentence-transformers,
- if local deps unavailable -> TF-IDF.

### In-memory vector store

`src/agentos/rag/vector_store.py` is intentionally simple:

- append-only document list with embeddings + metadata,
- cosine similarity scoring,
- threshold filtering + top-k retrieval,
- JSON save/load for persistence.

This makes retrieval mechanics easy to inspect before upgrading to external vector DBs.

### Drift detection with MMD

`src/agentos/rag/drift.py` implements embedding drift checks:

- stores reference embedding distribution,
- computes current distribution stats,
- uses kernel MMD (RBF with median heuristic gamma),
- reports drift if `mmd_score > threshold`,
- includes mean-direction cosine shift for interpretability.

MMD is chosen because it captures broader distribution shift than mean-only metrics.

## 4. Governance Design

### Defense-in-depth budget controls

`BudgetGuard` (`src/agentos/governance/budget.py`) stacks four limits:

- per-action,
- hourly,
- daily,
- lifetime total.

Any violated layer blocks execution; this prevents single-threshold misconfiguration from causing runaway spend.

### Why pre-check instead of post-check

Budget enforcement is intentionally split:

- `check_action(cost_estimate)` **before** call: fail fast, avoid spending.
- `record_spend(actual_cost)` **after** call: capture real usage.

This reduces wasted tokens and avoids double-counting.

### Immutable audit trail

`AuditLog` (`src/agentos/governance/audit.py`) records:

- timestamped action,
- allow/deny decision,
- rule and reason,
- contextual details.

Entries are append-only in memory and exportable as JSON for downstream compliance workflows.

### Kill switch implementation

`GovernanceEngine` (`src/agentos/governance/guardrails.py`) checks kill switch first in `check_tool_call`.

- If killed, all tool calls are blocked immediately.
- Action is logged in audit trail with `governance_rule="kill_switch"`.
- `kill(reason)` and `revive()` expose explicit operational controls for incident response.

## 5. Performance

Current benchmark claims are documented in `docs/benchmarks.md`:

- Combined evaluation ensemble (`overall_score`) reports `0.562` Spearman correlation with human judgment (N=50).
- Embedding similarity alone reports `0.600` Spearman correlation with human judgment (N=50).
- Full governance stack adds `0.02` ms median / `0.03` ms P95 overhead (measured over 10000 iterations).

These values should be treated as reproducible baselines and periodically revalidated as providers/models evolve.

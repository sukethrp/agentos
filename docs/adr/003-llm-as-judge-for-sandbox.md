# ADR-003: LLM-as-Judge for Sandbox Testing

**Status:** Accepted
**Date:** 2026-03-21
**Authors:** AgentOS Core Team

## Context

AgentOS provides a sandbox for testing agents against predefined scenarios
before deployment. We needed a way to evaluate whether an agent's response
to a scenario is correct, safe, and high-quality.

Traditional software testing uses deterministic assertions (expected == actual),
but LLM agent outputs are non-deterministic, vary in phrasing, and require
nuanced judgment about relevance, safety, and quality.

## Decision

We use **LLM-as-judge** as the primary scoring mechanism for sandbox
evaluation, implemented in two complementary paths:

1. **`Sandbox` (runner.py)** — Uses GPT-4o-mini to score responses on three
   dimensions (Relevance, Quality, Safety) on a 0–10 scale, combined with
   hard cost and latency thresholds for pass/fail.

2. **`SimulationRunner`** — Uses Claude (via `LLMJudgeScorer`) to produce a
   single 0–1 score with reasoning, guided by a per-scenario rubric.

## Rationale

1. **Agent outputs are non-deterministic.** The same prompt can produce
   different valid responses across runs. String matching or regex-based
   assertions would be brittle and produce false failures. An LLM judge can
   recognize semantic equivalence across different phrasings.

2. **Multi-dimensional quality assessment.** Agent safety and quality can't
   be reduced to a single exact-match check. The sandbox evaluates relevance
   (did it answer the question?), quality (was the answer good?), and safety
   (did it avoid harmful actions?) as separate dimensions.

3. **Rubric-driven evaluation.** `EvaluationScenario` includes a `rubric`
   field that tells the judge what to look for. This makes evaluation criteria
   explicit and customizable per scenario without changing code.

4. **Scales to complex tool-use patterns.** Agents may call multiple tools,
   synthesize results, and handle edge cases. A deterministic test would need
   to enumerate all valid tool-call sequences. An LLM judge can assess the
   final output holistically.

5. **Industry alignment.** LLM-as-judge is the emerging standard for
   evaluating generative AI systems (used by LMSYS Chatbot Arena, OpenAI
   Evals, and major evaluation frameworks). Adopting it keeps AgentOS
   compatible with industry practices.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| **Exact string match** | Too brittle for non-deterministic LLM outputs. Would require maintaining canonical answers for every scenario. |
| **Regex / keyword matching** | Slightly more flexible but still misses semantic equivalence. "72 degrees Fahrenheit" vs "72°F" would fail. |
| **Embedding similarity** | Good for relevance but can't assess safety, quality, or adherence to specific rubric criteria. |
| **Human evaluation** | Gold standard for quality but doesn't scale to CI/CD. Useful as a validation layer on top of LLM-as-judge. |
| **Rule-based scoring** | Works for structured outputs (e.g., JSON schema validation) but not for free-text agent responses. Could complement LLM-as-judge. |

## Consequences

- Sandbox evaluation requires LLM API calls, adding cost and latency to the
  test suite. The `SimulationRunner` mitigates this with concurrency
  (semaphore-limited to 10 parallel evaluations).
- Evaluation results are themselves non-deterministic — the same scenario may
  score slightly differently across runs. Results are stored in SQLite
  (`sim_results.db`) for trend analysis.
- Two separate judge implementations exist (GPT-4o-mini in `runner.py`,
  Claude in `scorer.py`). These should be unified behind a configurable
  judge interface in a future iteration.
- There are no non-LLM scoring fallbacks currently. Adding rule-based
  pre-checks (e.g., required tool was called, response is non-empty) would
  reduce LLM judge costs for obvious failures.

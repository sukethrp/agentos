# ADR-003: LLM-as-Judge Testing

## Status: Accepted

## Date: 2025-02-18

## Context

We need automated quality scoring for agent responses in the sandbox environment. Manual review doesn't scale, and traditional pattern-matching approaches can't capture the nuance of natural language quality.

## Decision

Use a separate LLM call to judge response quality on a 0–10 scale. The judge evaluates agent outputs against expected criteria including correctness, relevance, completeness, and tone.

## Alternatives Considered

- **Regex/rule-based scoring** — Too brittle for natural language; breaks on paraphrasing, formatting changes, or any response that doesn't match exact patterns.
- **Human review** — Gold standard for quality but doesn't scale for continuous testing during development and CI/CD pipelines.
- **Embedding similarity** — Captures semantic closeness but misses important nuances like factual correctness, instruction-following, and response structure.

## Consequences

- More expensive per test than deterministic approaches, but the cost is negligible when using gpt-4o-mini as the judge model.
- Scores correlate well with human judgment in our benchmarks, providing confidence in automated quality gates.
- Enables continuous quality regression testing as part of the development workflow.
- Judge prompts need to be maintained and calibrated alongside the agents they evaluate.

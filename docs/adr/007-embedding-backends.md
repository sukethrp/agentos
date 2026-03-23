# ADR-007: Multiple Embedding Backends for RAG

- Status: Accepted
- Date: 2026-03-23

## Context

The RAG pipeline previously depended only on OpenAI embeddings. That works for
rapid prototyping but is limiting for production:

- Some deployments cannot send data to third-party APIs due to privacy and
  compliance requirements.
- API-only embeddings add recurring cost and external latency.
- Teams need fallback options for offline development and zero-key demos.

## Decision

We introduce a backend abstraction in `src/agentos/rag/embeddings.py` with:

1. `OpenAIEmbeddings` (default when `OPENAI_API_KEY` is set)
2. `LocalEmbeddings` via Sentence Transformers (`all-MiniLM-L6-v2`)
3. `TFIDFEmbeddings` baseline via scikit-learn (fallback when local model deps
   are unavailable)

Selection behavior:

- Use OpenAI when `OPENAI_API_KEY` is present.
- Otherwise try local Sentence Transformers.
- If local dependencies are unavailable, use TF-IDF.

## Rationale

This keeps the platform usable across three common operating modes:

- Managed cloud setup (OpenAI quality, no local model maintenance)
- Privacy-sensitive deployment (local embeddings, no external calls)
- Minimal environments (TF-IDF fallback, no model download required)

Sentence Transformers such as `all-MiniLM-L6-v2` are close to hosted embedding
quality for many RAG tasks while removing API spend and reducing data exposure.

## Consequences

Positive:

- Better portability across local, air-gapped, and cloud environments.
- Lower operational cost for development and internal deployments.
- Stronger interview and architecture signal: API + local + classical baseline.

Trade-offs:

- Additional optional dependencies for local mode.
- TF-IDF quality is lower than neural embeddings for semantic matching.

## Benchmark Notes

This ADR tracks relative quality tiers rather than fixed benchmark claims:

- OpenAI embeddings are the strongest default in many semantic retrieval tasks.
- `all-MiniLM-L6-v2` is typically close enough for most production RAG.
- TF-IDF remains useful as a transparent baseline and cold-start fallback.

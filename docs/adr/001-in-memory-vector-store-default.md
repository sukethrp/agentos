# ADR-001: In-Memory Vector Store as Default

**Status:** Accepted
**Date:** 2026-03-21
**Authors:** AgentOS Core Team

## Context

AgentOS includes a RAG (Retrieval-Augmented Generation) pipeline that needs a
vector store for embedding storage and similarity search. The ecosystem offers
many options — Pinecone, Weaviate, ChromaDB, pgvector — each with different
operational requirements.

We needed to decide what ships as the default and what requires opt-in.

## Decision

We ship an **in-memory `VectorStore`** (with optional JSON persistence) as the
zero-dependency default, and support ChromaDB, Pinecone, and pgvector as
pluggable backends behind the `pip install agentos-platform[rag]` extra.

The store selection flows through `rag_config["vector_store"]` and the
`_get_store()` factory in the ingestion pipeline.

## Rationale

1. **Zero-setup onboarding.** New users can `pip install agentos-platform` and
   run RAG demos without provisioning a database, creating cloud accounts, or
   managing Docker containers. Reducing friction to first-run matters for
   adoption.

2. **No external service dependency.** Pinecone requires an API key and network
   access. pgvector requires a running PostgreSQL instance. Even ChromaDB needs
   a native dependency (`hnswlib`). The in-memory store uses only Python
   stdlib + NumPy-free cosine similarity.

3. **Sufficient for prototyping.** Most agent development starts with tens to
   hundreds of documents. In-memory cosine similarity is fast enough at this
   scale and avoids premature infrastructure decisions.

4. **Clear upgrade path.** The `BaseVectorStore` interface (`add`, `add_batch`,
   `search`, `size`) is implemented by all backends. Switching from in-memory
   to ChromaDB or Pinecone requires only changing `rag_config["vector_store"]`
   and installing the optional dependency — no code changes to the pipeline.

5. **JSON persistence for durability.** The in-memory store supports `save()`
   and `load()` for simple file-based persistence, which is adequate for
   single-process use during development.

## Alternatives Considered

| Alternative | Why not default |
|-------------|-----------------|
| **ChromaDB** | Requires native `hnswlib` dependency; build failures on some platforms. Used as the recommended first upgrade. |
| **Pinecone** | Cloud-only, requires API key, network latency. Not suitable for offline or local-first development. |
| **pgvector** | Requires PostgreSQL server. Significant operational overhead for getting started. |
| **FAISS** | C++ dependency with complex build matrix. Overkill for prototype-scale data. |

## Consequences

- Developers get a working RAG pipeline out of the box with no setup.
- In-memory store does not scale beyond ~10K documents in a single process.
- `RAGPipeline` currently always uses the in-memory store; it should be updated
  to respect `rag_config` for consistency with `IngestionPipeline`.
- Documentation must clearly communicate when to upgrade to a production store.

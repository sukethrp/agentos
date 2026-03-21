# ADR-001: In-Memory Vector Store

## Status: Accepted

## Date: 2025-02-10

## Context

The RAG pipeline needs vector storage for document embeddings. We need a solution that lets agents retrieve relevant context from ingested documents using semantic similarity search.

## Decision

Use an in-memory vector store with cosine similarity for storing and querying document embeddings.

## Alternatives Considered

- **Pinecone** — Managed vector database with excellent performance, but introduces vendor lock-in and ongoing cost that doesn't make sense at our current scale.
- **ChromaDB** — Solid open-source option, but adds an extra dependency and requires its own process/configuration for persistence.
- **Weaviate** — Feature-rich vector DB, but brings significant infrastructure overhead and operational complexity for what we need today.

## Consequences

- Simple, zero-config setup — no external services or infrastructure required.
- Works well for small-to-medium datasets that fit in memory.
- Fast iteration during development since there's nothing to provision or manage.
- Large datasets will eventually exceed memory limits and need an external vector DB — we plan to add storage adapters later to support pluggable backends.

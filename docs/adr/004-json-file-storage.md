# ADR-004: JSON File Storage

## Status: Accepted

## Date: 2025-02-22

## Context

We need persistence for authentication data, usage tracking, and marketplace metadata. The storage solution should require zero infrastructure setup and be easy to inspect during development.

## Decision

Use JSON file storage for all persistent data, with no external database required.

## Alternatives Considered

- **SQLite** — Good embedded option with real query capabilities, but adds complexity in schema management and migrations for data that is naturally document-shaped.
- **PostgreSQL** — Production-grade relational database, but overkill for a single-instance platform and requires external infrastructure.
- **Redis** — Fast key-value store, but data is ephemeral by default and requires a separate process.

## Consequences

- Zero-config setup — files are created on first write, no provisioning needed.
- Human-readable storage makes debugging straightforward; you can inspect and edit data with any text editor.
- Simple backup and version control — data files can be copied or even committed to a repo for reproducibility.
- Not suitable for high-concurrency production workloads due to file locking limitations — we will add database adapters in future versions to support scaling.

# ADR-004: JSON File Storage Over a Database

**Status:** Accepted
**Date:** 2026-03-21
**Authors:** AgentOS Core Team

## Context

AgentOS needs to persist several types of data: agent key-value state,
marketplace metadata, compliance audit logs, and simulation results. We needed
to choose a storage strategy that balances simplicity, portability, and
production-readiness.

## Decision

We default to **JSON files** and **in-memory stores** for most persistence,
with **SQLite** available as an opt-in backend for core storage and simulation
results. No external database is required to run AgentOS.

The storage landscape across components:

| Component | Default Format | Persistence |
|-----------|---------------|-------------|
| Core `AgentStorage` | JSON file (one per agent) | Yes |
| Marketplace `MarketplaceStore` | Single JSON file | Yes |
| Compliance `AuditLogger` | JSON Lines (append-only) | Yes |
| `SimulationRunner` results | SQLite | Yes |
| Monitor `AgentStore` | In-memory | No |
| Governance `AuditLog` | In-memory (export to JSON) | No |

## Rationale

1. **Zero infrastructure requirement.** AgentOS targets developers who want
   to prototype AI agents quickly. Requiring PostgreSQL, Redis, or MongoDB
   before writing the first agent would kill adoption. JSON files work
   everywhere Python runs.

2. **Human-readable and debuggable.** JSON files can be opened in any editor,
   piped through `jq`, or committed to version control. When debugging why an
   agent behaved a certain way, being able to `cat agent_data/my-agent.json`
   is invaluable.

3. **Portable across environments.** JSON files move between local dev,
   Docker containers, and CI without connection strings, migrations, or
   schema management. `cp -r agent_data/ backup/` is a complete backup.

4. **Appropriate scale.** AgentOS is in alpha (v0.3.x). Current usage
   patterns involve single-user development with tens of agents and thousands
   of events — well within what file-based storage handles comfortably.

5. **SQLite where needed.** The `SimulationRunner` stores results in SQLite
   because evaluation runs produce structured, queryable data (scores per
   scenario per run). Core `AgentStorage` also supports a SQLite backend for
   users who need concurrent access. This keeps the door open without
   mandating it.

6. **JSON Lines for compliance.** The audit logger uses append-only JSON Lines
   (`audit.log`) — each event is one line of JSON. This is a common pattern
   for audit trails: append-only, streamable, and parseable line-by-line
   without loading the full file.

## Alternatives Considered

| Alternative | Why not default |
|-------------|-----------------|
| **PostgreSQL** | Requires a running server, connection management, migrations. Overkill for prototype-phase storage. |
| **SQLite everywhere** | Better than Postgres for zero-setup, but binary format loses human-readability. Used where queryability matters (simulation results). |
| **Redis** | In-memory with optional persistence. Adds a service dependency. Better suited for caching and pub/sub (may be added for monitoring). |
| **MongoDB** | Document store fits the JSON-like data model, but adds a heavy service dependency and driver. |

## Consequences

- New users get working persistence with zero configuration.
- JSON files do not support concurrent writes safely. Multi-process
  deployments should use the SQLite backend for `AgentStorage`.
- In-memory stores (monitor, governance audit) lose data on restart. This is
  acceptable for development but production deployments will need persistent
  alternatives.
- No query language for JSON files — filtering events requires loading the
  full file. For the compliance audit log, this becomes a problem at scale
  and should be migrated to SQLite or a log aggregation service.
- The storage abstraction (`AgentStorage` with JSON/SQLite/memory backends)
  provides a pattern for adding new backends without changing consuming code.

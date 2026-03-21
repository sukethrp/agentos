# Architecture Decision Records

This directory documents key design decisions in AgentOS using the
[ADR format](https://adr.github.io/). Each record captures the context,
decision, rationale, alternatives considered, and consequences.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-in-memory-vector-store-default.md) | In-Memory Vector Store as Default | Accepted |
| [002](002-fastapi-over-flask-django.md) | FastAPI Over Flask/Django | Accepted |
| [003](003-llm-as-judge-for-sandbox.md) | LLM-as-Judge for Sandbox Testing | Accepted |
| [004](004-json-file-storage-over-database.md) | JSON File Storage Over a Database | Accepted |
| [005](005-governed-agent-class-over-decorators.md) | Single GovernedAgent Class Over Separate Decorators | Accepted |

## Contributing

When making a significant architectural decision, add a new ADR:

1. Copy the template: `NNN-short-title.md`
2. Fill in: **Context**, **Decision**, **Rationale**, **Alternatives Considered**, **Consequences**
3. Set status to `Proposed` and open a PR for discussion
4. Update status to `Accepted` or `Rejected` after review

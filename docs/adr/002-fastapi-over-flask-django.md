# ADR-002: FastAPI Over Flask/Django

**Status:** Accepted
**Date:** 2026-03-21
**Authors:** AgentOS Core Team

## Context

AgentOS needs a web framework for its platform UI, REST API, real-time
monitoring dashboard, and streaming chat interface. The Python ecosystem
offers three mature options: Flask, Django, and FastAPI.

## Decision

We chose **FastAPI** as the web framework for all AgentOS web components
(platform app, monitoring server, and API routers).

## Rationale

1. **Native WebSocket support.** AgentOS has two critical real-time channels:
   `/ws/chat` for streaming agent responses token-by-token, and `/ws/monitor`
   for broadcasting live tool-call and cost events to the dashboard. FastAPI's
   first-class `WebSocket` support (inherited from Starlette) makes this
   straightforward. Flask requires `flask-socketio` with additional
   dependencies; Django requires `channels` with an ASGI adapter.

2. **Async-first / ASGI.** Agent tool execution runs in `asyncio.to_thread`
   and tool calls are gathered with `asyncio.gather` for parallel execution.
   The web layer needs to be async-compatible to avoid blocking during these
   operations. FastAPI runs on ASGI (via uvicorn) natively, while Flask's WSGI
   model requires workarounds for async patterns.

3. **Pydantic integration.** AgentOS already uses Pydantic extensively for its
   core types (`AgentConfig`, `ToolSpec`, `AgentEvent`, `Message`, etc.).
   FastAPI validates request bodies using the same Pydantic models
   (`RunRequest`, `CreateScenarioRequest`, `ABTestRequest`), eliminating
   duplicate validation logic. Flask and Django would require separate
   serialization layers.

4. **Dependency injection.** Auth is handled via `Depends(get_current_user)`
   and `Depends(get_optional_user)`, which compose cleanly across routers
   without global middleware state. This pattern is built into FastAPI.

5. **Automatic API documentation.** FastAPI generates OpenAPI (Swagger) docs
   at `/docs` with zero configuration. This is valuable for an open-source
   project where contributors need to understand the API surface quickly.

6. **Lightweight, no ORM.** AgentOS uses JSON files and in-memory stores
   rather than a relational database (see ADR-004). Django's ORM, admin panel,
   and migrations system would be unused overhead.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| **Flask** | WSGI-only (no native async/WebSocket), would need flask-socketio, no built-in validation, no auto-docs without extensions. |
| **Django** | Heavy ORM and admin we don't need, WSGI by default, requires `channels` for WebSocket, Pydantic integration is bolt-on. |
| **Starlette** | FastAPI is built on Starlette and adds validation + docs. Using raw Starlette would mean reimplementing those features. |

## Consequences

- All web endpoints are async-capable and WebSocket-ready out of the box.
- Contributors familiar with FastAPI (increasingly common in the AI/ML
  ecosystem) can onboard quickly.
- The framework is tightly coupled to Pydantic v2 — a major Pydantic version
  change would require coordinated updates across core types and API layer.
- The monitoring server (`monitor/server.py`) runs as a separate FastAPI app
  on its own port, which is simple but means two processes in production.

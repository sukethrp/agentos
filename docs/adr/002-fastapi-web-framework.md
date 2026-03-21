# ADR-002: FastAPI Web Framework

## Status: Accepted

## Date: 2025-02-12

## Context

We need a web framework for the platform API and UI. The framework must support asynchronous request handling, WebSockets for real-time agent communication, and automatic API documentation for the marketplace and developer experience.

## Decision

Use FastAPI as the web framework for the AgentOS platform.

## Alternatives Considered

- **Flask** — Mature and widely adopted, but lacks native async support and automatic API documentation generation.
- **Django** — Full-featured framework, but too heavy for our use case — we don't need its ORM, admin panel, or templating engine.
- **Starlette** — FastAPI is built on Starlette, but using it directly is too low-level and would require us to reimplement routing, validation, and docs that FastAPI provides out of the box.

## Consequences

- Automatic OpenAPI documentation for every endpoint, reducing maintenance burden and improving developer experience.
- Native async/await support for non-blocking I/O, critical for LLM API calls and concurrent agent execution.
- Built-in WebSocket support for real-time agent streaming and monitoring.
- Pydantic integration for request/response validation matches our existing type system and governance models.
- Large ecosystem and community for middleware, authentication, and tooling.

# ADR-005: Governed Agent Unified Class

## Status: Accepted

## Date: 2025-02-26

## Context

Users needed to manually compose budget controls, permission guards, and monitoring when creating governed agents. This led to boilerplate-heavy setup code and inconsistent governance patterns across different agent implementations.

## Decision

Provide a single `GovernedAgent` class that wraps all governance features — budget enforcement, permission checking, and usage monitoring — into one unified interface.

## Alternatives Considered

- **Decorator pattern** — Stacking `@budget`, `@permissions`, `@monitor` decorators on agent methods is too implicit; the governance behavior is hidden and the execution order is hard to reason about.
- **Middleware chain** — Configuring a pipeline of governance middleware adds complex configuration and makes it difficult to understand which checks run when.
- **Mixin classes** — Multiple inheritance with `BudgetMixin`, `PermissionMixin`, etc. leads to diamond inheritance issues and fragile method resolution order.

## Consequences

- Simple ~10-line setup to get a fully governed agent with budget, permissions, and monitoring.
- All governance configuration is visible in one place, making it easy to audit and understand.
- Power users can still use individual guard classes directly for custom compositions.
- Adding new governance features means extending the `GovernedAgent` class, which keeps the API surface small and discoverable.

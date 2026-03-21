# ADR-005: Single GovernedAgent Class Over Separate Decorators

**Status:** Accepted
**Date:** 2026-03-21
**Authors:** AgentOS Core Team

## Context

AgentOS provides governance features — budget limits, permission controls,
kill switches, audit logging, and sandbox testing. We needed to decide how
to expose these to users: as composable decorators/middleware that wrap
individual tools or agent calls, or as a single unified class.

## Decision

We implement a **single `GovernedAgent` class** that composes `Agent`,
`GovernanceEngine`, monitor integration, and `Sandbox` into one facade.

```python
agent = GovernedAgent(
    name="my-agent",
    model="gpt-4o-mini",
    tools=[calculator, weather],
    budget=BudgetGuard(max_per_day=5.00),
    permissions=PermissionGuard(blocked_tools=["dangerous_tool"]),
)

agent.run("What's 2 + 2?")   # governed execution
agent.test(scenarios)          # sandbox testing
agent.kill("emergency")        # kill switch
agent.audit()                  # view audit log
```

Governance is applied by wrapping each tool's `execute` method at
construction time, routing every call through the `GovernanceEngine`.

## Rationale

1. **Single entry point for users.** Governance, monitoring, and testing are
   inherently per-agent concerns. A single class gives users one thing to
   instantiate and one place to configure all policies. Compare:

   ```python
   # GovernedAgent (current)
   agent = GovernedAgent(name="a", tools=[t], budget=budget, permissions=perms)

   # Decorator alternative (rejected)
   agent = Agent(name="a", tools=[t])
   agent = with_budget(agent, budget)
   agent = with_permissions(agent, perms)
   agent = with_audit(agent)
   agent = with_sandbox(agent)
   ```

   The decorator version is more code, has ordering concerns, and makes it
   unclear what the final object's API surface is.

2. **Shared governance state.** `BudgetGuard` tracks cumulative spend across
   all tool calls. `PermissionGuard` tracks action counts per run. The
   `GovernanceEngine` orchestrates these guards in a specific order (kill
   switch -> permissions -> org caps -> budget -> audit). This stateful,
   ordered pipeline is naturally expressed as class composition rather than
   stacked decorators.

3. **Tool wrapping at construction time.** `GovernedAgent._wrap_tools()`
   monkey-patches `Tool.execute` with a closure that captures `self.governance`.
   This ensures governance checks happen transparently on every tool call,
   including those triggered by the LLM during multi-step reasoning. Decorators
   on the `Agent.run()` method would miss individual tool-level governance.

4. **Unified control surface.** `GovernedAgent` exposes `run()`, `test()`,
   `kill()`, `revive()`, `status()`, and `audit()` as a coherent API. With
   decorators, these methods would be scattered across wrapper objects or
   require a separate control-plane mechanism.

5. **Sandbox integration.** `test(scenarios)` runs the same governed agent
   through sandbox scenarios. This tight coupling between the agent under test
   and the governance configuration ensures that sandbox results reflect
   production behavior. A decorator approach would need to reconstruct the
   full decoration chain for the sandbox runner.

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| **Stacked decorators** | Ordering-sensitive, scattered API surface, hard to pass shared governance state between layers. |
| **Middleware pipeline** | Better than decorators for ordering, but still requires a separate composition step and obscures the final object's interface. |
| **Mixin classes** | `class GovernedAgent(Agent, BudgetMixin, PermissionMixin, ...)` — Python's MRO makes method resolution unpredictable with multiple mixins. Diamond inheritance with stateful mixins is fragile. |
| **Agent subclass only** | Governance as overridden methods in an Agent subclass. Doesn't compose — you'd need `BudgetAgent`, `PermissionAgent`, `BudgetPermissionAgent`, etc. Combinatorial explosion. |

## Consequences

- Users get a batteries-included governed agent with minimal configuration.
- The `GovernedAgent` class is tightly coupled to `GovernanceEngine` internals.
  Changes to governance ordering or new guard types require updating the class.
- Tool wrapping via monkey-patching `Tool.execute` is effective but means
  governance cannot be removed from a tool once applied. Creating a fresh
  `GovernedAgent` is the reset mechanism.
- Users who want only budget tracking or only permissions without the full
  `GovernedAgent` can use `BudgetGuard` and `PermissionGuard` directly —
  the guards are independent, composable objects. `GovernedAgent` is the
  opinionated high-level API.

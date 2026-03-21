# ADR-006: Agent Mesh Protocol for Multi-Agent Communication

**Status:** Accepted  
**Date:** 2026-03-21  
**Author:** AgentOS Team

## Context

AgentOS supports single-agent workflows but many real-world tasks benefit
from multiple specialist agents collaborating: a researcher gathers data,
a writer drafts a report, and a reviewer checks quality.  Users need a way
to compose agents into coordinated teams.

Key requirements:
- Named agents that can discover each other
- Subtask delegation with clear request/response semantics
- Shared context so agents build on each other's work
- Aggregated cost tracking across the full chain
- Support both top-down orchestration and lateral peer-to-peer patterns

## Decision

Implement a **mesh protocol** in `src/agentos/mesh/` with three components:

1. **AgentRegistry** — a thread-safe lookup table of named agents.
2. **AgentMesh** — the coordination layer.  Supports two patterns:
   - *Orchestrator*: a coordinator agent receives a `delegate` tool and the
     LLM decides when/whom to delegate to.
   - *Peer-to-peer*: every agent in the mesh gets a `delegate` tool.
3. **SharedContext** — a thread-safe key-value store injected into each
   agent's system prompt so downstream agents see upstream results.

Delegation is implemented as a **tool call**: when the LLM decides to
delegate, it calls the `delegate(agent_name, task)` tool.  The mesh
executes the target agent synchronously and returns the text result.

## Rationale

- **Tool-based delegation** keeps the interface natural for the LLM — no
  special protocol, just another tool call — and reuses the existing
  retry/timeout/caching infrastructure.
- **Orchestrator + P2P** covers the two most common multi-agent patterns
  without forcing users into either.
- **SharedContext** avoids the complexity of a full message bus while still
  enabling implicit coordination (e.g., researcher stores findings, writer
  reads them).
- **MeshCostTracker** aggregates per-agent costs so users can see the true
  cost of a multi-agent workflow.

## Alternatives Considered

| Alternative | Why Not |
|---|---|
| Async message queue (Redis/RabbitMQ) | Heavy infrastructure; overkill for in-process agent coordination |
| LangGraph / CrewAI integration | Vendor lock-in; we want a lightweight built-in primitive |
| Agent subclassing (OrchestratorAgent) | Breaks composition; tool-based delegation is more flexible |
| Shared vector store for context | Too slow for turn-by-turn coordination; key-value is sufficient |

## Consequences

**Positive:**
- Users can build multi-agent workflows with 5-10 lines of code
- Cost visibility across the entire agent chain
- SharedContext enables agents to build on each other's output
- No new dependencies — pure Python + existing AgentOS primitives

**Negative:**
- Delegation is synchronous (target agent runs to completion before returning);
  async streaming delegation is a future enhancement
- Circular delegation (A → B → A) is possible; users must design prompts to
  avoid infinite loops (a max-depth guard is a planned follow-up)
- SharedContext is in-memory only; persistence requires external storage

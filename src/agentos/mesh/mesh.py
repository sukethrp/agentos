"""AgentMesh — multi-agent coordination with orchestrator and peer-to-peer patterns.

Two patterns are supported:

1. **Orchestrator**: One coordinator agent delegates subtasks to specialist agents.
   The coordinator receives a ``delegate`` tool that lets the LLM decide which
   agent to call and what to ask.

2. **Peer-to-peer**: Any agent in the mesh can delegate to any other agent.
   Each agent receives a ``delegate`` tool scoped to the mesh.

Usage::

    from agentos.mesh import AgentMesh

    mesh = AgentMesh(name="research-team")
    mesh.add(researcher)
    mesh.add(writer)
    mesh.add(reviewer)

    # Orchestrator pattern — one coordinator drives the workflow
    result = mesh.run_orchestrated(
        coordinator=researcher,
        task="Research and write a report on AI safety",
    )

    # Peer-to-peer — each agent can delegate to any other
    mesh.enable_peer_delegation()
    result = researcher.run("Write a report, delegate writing to writer")
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from agentos.core.agent import Agent
from agentos.core.tool import Tool
from agentos.core.types import Message
from agentos.logging import get_logger
from agentos.mesh.protocol import (
    AgentMessage,
    DelegationRequest,
    DelegationResult,
    SharedContext,
)
from agentos.mesh.registry import AgentRegistry

_log = get_logger("agentos.mesh")


class MeshCostTracker:
    """Aggregate cost/token tracking across an entire agent chain."""

    def __init__(self) -> None:
        self.delegations: list[DelegationResult] = []
        self.messages: list[AgentMessage] = []

    @property
    def total_cost(self) -> float:
        return sum(d.cost_usd for d in self.delegations)

    @property
    def total_tokens(self) -> int:
        return sum(d.tokens_used for d in self.delegations)

    @property
    def total_latency_ms(self) -> float:
        return sum(d.latency_ms for d in self.delegations)

    def record(self, result: DelegationResult) -> None:
        self.delegations.append(result)

    def summary(self) -> dict[str, Any]:
        per_agent: dict[str, dict[str, Any]] = {}
        for d in self.delegations:
            key = d.to_agent
            if key not in per_agent:
                per_agent[key] = {"cost_usd": 0.0, "tokens": 0, "calls": 0}
            per_agent[key]["cost_usd"] += d.cost_usd
            per_agent[key]["tokens"] += d.tokens_used
            per_agent[key]["calls"] += 1

        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "total_delegations": len(self.delegations),
            "per_agent": per_agent,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'=' * 60}")
        print("Mesh Run Summary")
        print(f"{'=' * 60}")
        print(f"   Delegations:  {s['total_delegations']}")
        print(f"   Total tokens: {s['total_tokens']:,}")
        print(f"   Total cost:   ${s['total_cost_usd']:.4f}")
        print(f"   Total time:   {s['total_latency_ms']:.0f}ms")
        if s["per_agent"]:
            print("   Per-agent:")
            for name, stats in s["per_agent"].items():
                print(
                    f"      {name}: {stats['calls']} call(s), "
                    f"${stats['cost_usd']:.4f}, {stats['tokens']} tokens"
                )
        print(f"{'=' * 60}")


class AgentMesh:
    """Multi-agent mesh supporting orchestrator and peer-to-peer delegation."""

    def __init__(self, name: str = "mesh") -> None:
        self.name = name
        self.registry = AgentRegistry()
        self.shared_context = SharedContext()
        self.cost_tracker = MeshCostTracker()
        self._run_id = str(uuid.uuid4())

    # ── Agent management ──

    def add(self, agent: Agent) -> AgentMesh:
        """Add an agent to the mesh. Returns self for chaining."""
        self.registry.register(agent)
        return self

    def remove(self, name: str) -> None:
        self.registry.unregister(name)

    @property
    def agents(self) -> list[str]:
        return self.registry.list_agents()

    # ── Delegation (core mechanism) ──

    def delegate(self, from_agent: str, to_agent: str, task: str) -> str:
        """Delegate a task from one agent to another and return the text result."""
        target = self.registry.get(to_agent)
        if target is None:
            available = ", ".join(self.agents)
            return f"Error: Agent '{to_agent}' not found. Available: {available}"

        request = DelegationRequest(
            from_agent=from_agent,
            to_agent=to_agent,
            task=task,
            parent_run_id=self._run_id,
        )

        _log.info(
            "mesh.delegate",
            extra={
                "mesh": self.name,
                "from": from_agent,
                "to": to_agent,
                "task": task[:120],
            },
        )

        ctx_fragment = self.shared_context.to_prompt_fragment()
        if ctx_fragment:
            augmented_task = f"{task}\n\n{ctx_fragment}"
        else:
            augmented_task = task

        start = time.time()
        response = target.run(augmented_task)
        latency_ms = (time.time() - start) * 1000

        result_text = response.content or ""
        total_cost = sum(e.cost_usd for e in target.events)
        total_tokens = sum(e.tokens_used for e in target.events)

        delegation_result = DelegationResult(
            request_id=request.id,
            from_agent=from_agent,
            to_agent=to_agent,
            result=result_text,
            cost_usd=total_cost,
            tokens_used=total_tokens,
            latency_ms=round(latency_ms, 1),
        )
        self.cost_tracker.record(delegation_result)

        self.cost_tracker.messages.append(
            AgentMessage(
                sender=from_agent,
                receiver=to_agent,
                content=task,
                parent_run_id=self._run_id,
            )
        )
        self.cost_tracker.messages.append(
            AgentMessage(
                sender=to_agent,
                receiver=from_agent,
                content=result_text[:500],
                parent_run_id=self._run_id,
            )
        )

        _log.info(
            "mesh.delegate.done",
            extra={
                "mesh": self.name,
                "from": from_agent,
                "to": to_agent,
                "cost": total_cost,
                "tokens": total_tokens,
                "latency_ms": round(latency_ms, 1),
            },
        )

        return result_text

    # ── Orchestrator pattern ──

    def run_orchestrated(self, coordinator: Agent, task: str) -> Message:
        """Run a task with *coordinator* as the orchestrator.

        The coordinator is given a ``delegate`` tool it can use to farm
        out subtasks to any other agent in the mesh. It can call delegate
        multiple times in a single run.
        """
        self._run_id = str(uuid.uuid4())
        self.cost_tracker = MeshCostTracker()

        if coordinator.config.name not in self.registry:
            self.add(coordinator)

        delegate_tool = self._build_delegate_tool(coordinator.config.name)
        original_tools = list(coordinator.tools)
        coordinator.tools = original_tools + [delegate_tool]
        coordinator._tool_map[delegate_tool.name] = delegate_tool

        try:
            result = coordinator.run(task)
        finally:
            coordinator.tools = original_tools
            coordinator._tool_map = {t.name: t for t in original_tools}

        coord_cost = sum(e.cost_usd for e in coordinator.events)
        coord_tokens = sum(e.tokens_used for e in coordinator.events)
        self.cost_tracker.record(
            DelegationResult(
                request_id="coordinator",
                from_agent="user",
                to_agent=coordinator.config.name,
                result=(result.content or "")[:200],
                cost_usd=coord_cost,
                tokens_used=coord_tokens,
            )
        )

        self.cost_tracker.print_summary()
        return result

    # ── Peer-to-peer pattern ──

    def enable_peer_delegation(self) -> None:
        """Give every agent in the mesh a ``delegate`` tool so any agent
        can call any other agent."""
        for name in self.agents:
            agent = self.registry.get(name)
            if agent is None:
                continue
            if any(t.name == "delegate" for t in agent.tools):
                continue
            delegate_tool = self._build_delegate_tool(name)
            agent.tools.append(delegate_tool)
            agent._tool_map[delegate_tool.name] = delegate_tool

    def disable_peer_delegation(self) -> None:
        """Remove the ``delegate`` tool from all agents."""
        for name in self.agents:
            agent = self.registry.get(name)
            if agent is None:
                continue
            agent.tools = [t for t in agent.tools if t.name != "delegate"]
            agent._tool_map = {t.name: t for t in agent.tools}

    # ── Internal helpers ──

    def _build_delegate_tool(self, caller_name: str) -> Tool:
        """Create a delegate tool bound to this mesh for a specific caller."""
        peers = [n for n in self.agents if n != caller_name]
        peer_list = ", ".join(peers) if peers else "(none yet)"

        def delegate(agent_name: str, task: str) -> str:
            """Delegate a subtask to another agent in the mesh.
            Provide the target agent name and the task description."""
            return self.delegate(caller_name, agent_name, task)

        return Tool(
            fn=delegate,
            name="delegate",
            description=(
                f"Delegate a subtask to another agent in the mesh. "
                f"Available agents: {peer_list}. "
                f"Provide the agent_name and a clear task description."
            ),
            timeout_seconds=120.0,
        )

    def reset(self) -> None:
        """Reset run state (cost tracker, shared context) for a fresh run."""
        self._run_id = str(uuid.uuid4())
        self.cost_tracker = MeshCostTracker()
        self.shared_context = SharedContext()

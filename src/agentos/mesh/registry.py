"""Agent Registry — global lookup table for named agents.

Enables agents to discover and communicate with each other by name.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentos.core.agent import Agent


class AgentRegistry:
    """Thread-safe registry of named agents.

    Usage::

        registry = AgentRegistry()
        registry.register(researcher_agent)
        registry.register(writer_agent)

        agent = registry.get("researcher")
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._lock = threading.Lock()

    def register(self, agent: Agent) -> None:
        with self._lock:
            self._agents[agent.config.name] = agent

    def unregister(self, name: str) -> None:
        with self._lock:
            self._agents.pop(name, None)

    def get(self, name: str) -> Agent | None:
        with self._lock:
            return self._agents.get(name)

    def list_agents(self) -> list[str]:
        with self._lock:
            return list(self._agents.keys())

    def clear(self) -> None:
        with self._lock:
            self._agents.clear()

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._agents

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    def __repr__(self) -> str:
        return f"AgentRegistry({self.list_agents()})"


# Singleton for convenience; meshes can also use their own instance.
default_registry = AgentRegistry()

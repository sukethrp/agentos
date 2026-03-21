"""Mesh protocol — message types and shared context for agent-to-agent communication."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """A message passed between agents in a mesh."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender: str
    receiver: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_run_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


class DelegationRequest(BaseModel):
    """A request from one agent to delegate a subtask to another."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    from_agent: str
    to_agent: str
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    parent_run_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


class DelegationResult(BaseModel):
    """The result of a delegated subtask."""

    request_id: str
    from_agent: str
    to_agent: str
    result: str
    cost_usd: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True


class SharedContext:
    """Thread-safe shared key-value store for agents in a mesh.

    All agents in a mesh share a single ``SharedContext`` instance.
    They can read/write facts that are visible to every other agent,
    enabling implicit coordination without direct message passing.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []

    def set(self, key: str, value: Any, *, author: str = "") -> None:
        with self._lock:
            self._store[key] = value
            self._history.append(
                {"key": key, "value": value, "author": author, "ts": time.time()}
            )

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def get_all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._store)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def to_prompt_fragment(self) -> str:
        """Render shared context as a text block for system-prompt injection."""
        with self._lock:
            if not self._store:
                return ""
            lines = ["[SHARED CONTEXT from other agents]"]
            for k, v in self._store.items():
                lines.append(f"- {k}: {v}")
            return "\n".join(lines)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"SharedContext({len(self)} keys)"

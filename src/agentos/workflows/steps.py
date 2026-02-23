"""Core building blocks for AgentOS workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StepResult:
    """Result of executing a single workflow step."""

    name: str
    output: str = ""
    status: str = "pending"  # pending|running|completed|failed|skipped|fallback
    cost: float = 0.0
    duration_ms: float = 0.0
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """A single unit of work in a workflow."""

    name: str
    agent: Any  # agentos.core.agent.Agent
    query: Any  # str | Callable[[Dict[str, StepResult]], str]
    when: Optional[str] = None
    when_not: Optional[str] = None
    max_retries: int = 0
    fallback_agent: Any = None
    fallback_query: Any = None


@dataclass
class Condition:
    """Named condition that can be referenced by later steps."""

    name: str
    predicate: Callable[[StepResult, Dict[str, StepResult]], bool]
    source_step: Optional[str] = None


@dataclass
class ParallelGroup:
    """Group of steps that should run in parallel."""

    name: str
    steps: List[Step] = field(default_factory=list)

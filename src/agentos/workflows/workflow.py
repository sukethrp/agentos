"""Fluent workflow definition API for AgentOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from agentos.workflows.steps import Step, Condition, ParallelGroup


@dataclass
class Workflow:
    """A multi-step agent workflow with optional branching and parallelism."""

    name: str
    steps: List[Any] = field(default_factory=list)  # Step | ParallelGroup
    conditions: Dict[str, Condition] = field(default_factory=dict)

    # ── Fluent API ──

    def __post_init__(self) -> None:
        self._last_step: Step | None = None

    def step(
        self,
        name: str,
        agent: Any,
        query: Any,
        when: str | None = None,
        when_not: str | None = None,
        max_retries: int = 0,
        fallback_agent: Any | None = None,
        fallback_query: Any | None = None,
    ) -> "Workflow":
        """Append a sequential step to the workflow.

        Args:
            name: Step identifier.
            agent: Agent instance to run.
            query: Either a string (optionally with .format placeholders)
                   or a callable taking (context: dict[str, StepResult]) -> str.
            when: Optional condition name that must evaluate to True to run.
            when_not: Optional condition name that must evaluate to False to run.
        """
        step = Step(
            name=name,
            agent=agent,
            query=query,
            when=when,
            when_not=when_not,
            max_retries=max_retries,
            fallback_agent=fallback_agent,
            fallback_query=fallback_query,
        )
        self.steps.append(step)
        self._last_step = step
        return self

    def condition(self, name: str, fn: Callable[[str], bool]) -> "Workflow":
        """Define a condition that can be referenced by later steps.

        The provided function receives the output string from the source step.
        Conditions are automatically associated with the last defined step.
        """
        if not self._last_step:
            raise ValueError("condition() must be called after at least one step()")

        source_step_name = self._last_step.name

        def predicate(result, all_results):
            # result is StepResult; we pass its output into the user function
            return bool(fn(result.output or ""))

        cond = Condition(name=name, predicate=predicate, source_step=source_step_name)
        self.conditions[name] = cond
        return self

    def parallel(
        self,
        name: str,
        steps: Sequence[tuple[str, Any, Any]],
    ) -> "Workflow":
        """Add a parallel group of steps.

        Args:
            name: Group name.
            steps: Sequence of (step_name, agent, query) tuples.
        """
        pg_steps = [Step(s_name, agent, query) for (s_name, agent, query) in steps]
        group = ParallelGroup(name=name, steps=pg_steps)
        self.steps.append(group)
        self._last_step = None
        return self

    def build(self) -> "Workflow":
        """Finalize the workflow (currently a no-op, returns self)."""
        return self


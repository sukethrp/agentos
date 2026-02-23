"""Workflow execution engine for AgentOS."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agentos.events import event_bus
from agentos.workflows.steps import Step, StepResult, ParallelGroup
from agentos.workflows.workflow import Workflow


@dataclass
class WorkflowExecution:
    """Audit trail for a single workflow execution."""

    id: str
    workflow_name: str
    started_at: float
    ended_at: float | None = None
    status: str = "running"  # running|completed|failed
    steps: Dict[str, StepResult] = field(default_factory=dict)
    path: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow_name": self.workflow_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "steps": {name: sr.__dict__ for name, sr in self.steps.items()},
            "path": self.path,
        }


class WorkflowRunner:
    """Execute a Workflow and emit events for each step completion."""

    def __init__(self, workflow: Workflow):
        self.workflow = workflow

    # ── Public API ──

    def run(self, context: Dict[str, Any] | None = None) -> WorkflowExecution:
        ctx: Dict[str, Any] = context.copy() if context else {}
        exec_id = uuid.uuid4().hex[:12]
        execution = WorkflowExecution(
            id=exec_id,
            workflow_name=self.workflow.name,
            started_at=time.time(),
        )

        results: Dict[str, StepResult] = {}
        last_output: str = ""

        try:
            for item in self.workflow.steps:
                if isinstance(item, Step):
                    res, last_output = self._run_step(
                        item, ctx, results, last_output, execution
                    )
                    results[res.name] = res
                    execution.steps[res.name] = res
                    if res.status not in ("completed", "fallback"):
                        # Fail-fast on hard failure
                        execution.status = "failed"
                        break
                elif isinstance(item, ParallelGroup):
                    group_results = self._run_parallel_group(
                        item, ctx, results, last_output, execution
                    )
                    for r in group_results:
                        results[r.name] = r
                        execution.steps[r.name] = r
                        if r.status not in ("completed", "fallback"):
                            execution.status = "failed"
                    # Do not update last_output from parallel steps,
                    # but expose their outputs via ctx[step_name].
                else:
                    continue

            if execution.status == "running":
                execution.status = "completed"
        finally:
            execution.ended_at = time.time()

        return execution

    # ── Internal helpers ──

    def _should_run(self, step: Step, results: Dict[str, StepResult]) -> bool:
        if step.when:
            cond = self.workflow.conditions.get(step.when)
            if cond:
                src_name = cond.source_step or step.when
                src_res = results.get(src_name)
                if not src_res:
                    return False
                return cond.predicate(src_res, results)
        if step.when_not:
            cond = self.workflow.conditions.get(step.when_not)
            if cond:
                src_name = cond.source_step or step.when_not
                src_res = results.get(src_name)
                if not src_res:
                    return True
                return not cond.predicate(src_res, results)
        return True

    def _build_query(
        self,
        step: Step,
        ctx: Dict[str, Any],
        results: Dict[str, StepResult],
        last_output: str,
    ) -> str:
        if callable(step.query):
            return step.query(results)

        # String template: provide access to last_output and per-step outputs
        variables: Dict[str, Any] = {
            "last_output": last_output,
        }
        for name, res in results.items():
            variables[name] = res.output
        variables.update(ctx)
        try:
            return str(step.query).format(**variables)
        except Exception:
            # Fallback to raw query string
            return str(step.query)

    def _execute_agent(self, agent: Any, query: str) -> StepResult:
        start = time.time()
        sr = StepResult(name=getattr(agent.config, "name", "step"), status="running")
        try:
            # Suppress agent's stdout during workflow run
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            msg = agent.run(query)
            sys.stdout = old_stdout

            output = getattr(msg, "content", "") or ""
            cost = sum(e.cost_usd for e in getattr(agent, "events", []))
            duration_ms = (time.time() - start) * 1000

            sr.output = output
            sr.status = "completed"
            sr.cost = float(cost)
            sr.duration_ms = round(duration_ms, 2)
        except Exception as e:
            sr.status = "failed"
            sr.error = str(e)
            sr.duration_ms = round((time.time() - start) * 1000, 2)
        return sr

    def _run_step(
        self,
        step: Step,
        ctx: Dict[str, Any],
        results: Dict[str, StepResult],
        last_output: str,
        execution: WorkflowExecution,
    ) -> tuple[StepResult, str]:
        """Run a single step with retries and optional fallback."""
        if not self._should_run(step, results):
            sr = StepResult(name=step.name, status="skipped")
            execution.path.append(f"{step.name} (skipped)")
            self._emit_event(execution, sr)
            return sr, last_output

        attempt = 0
        final_result: StepResult | None = None

        while attempt <= step.max_retries:
            query = self._build_query(step, ctx, results, last_output)
            sr = self._execute_agent(step.agent, query)
            sr.name = step.name
            attempt += 1
            if sr.status == "completed":
                final_result = sr
                break

        if final_result is None and step.fallback_agent:
            # Run fallback once
            fb_query = step.fallback_query or step.query
            fb_step = Step(
                name=step.name,
                agent=step.fallback_agent,
                query=fb_query,
            )
            sr = self._execute_agent(
                fb_step.agent, self._build_query(fb_step, ctx, results, last_output)
            )
            sr.name = step.name
            if sr.status == "completed":
                sr.status = "fallback"
            final_result = sr

        if final_result is None:
            final_result = StepResult(
                name=step.name, status="failed", error="All retries and fallback failed"
            )

        # Update context and path
        if final_result.status in ("completed", "fallback"):
            ctx[step.name] = final_result.output
            last_output = final_result.output
        execution.path.append(step.name)

        self._emit_event(execution, final_result)
        return final_result, last_output

    def _run_parallel_group(
        self,
        group: ParallelGroup,
        ctx: Dict[str, Any],
        results: Dict[str, StepResult],
        last_output: str,
        execution: WorkflowExecution,
    ) -> List[StepResult]:
        group_results: List[StepResult] = []
        threads: List[threading.Thread] = []
        lock = threading.Lock()

        def run_single(s: Step):
            nonlocal group_results
            r, _ = self._run_step(s, ctx, results, last_output, execution)
            with lock:
                group_results.append(r)

        for s in group.steps:
            t = threading.Thread(target=run_single, args=(s,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return group_results

    def _emit_event(self, execution: WorkflowExecution, result: StepResult) -> None:
        """Emit workflow.step.completed event via the global event bus."""
        try:
            event_bus.emit(
                "workflow.step.completed",
                data={
                    "workflow_id": execution.id,
                    "workflow_name": execution.workflow_name,
                    "step_name": result.name,
                    "status": result.status,
                    "cost": result.cost,
                    "duration_ms": result.duration_ms,
                    "error": result.error,
                },
                source="workflow_runner",
            )
        except Exception:
            # Emission failures should not break the workflow
            pass

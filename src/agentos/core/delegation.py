from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from agentos.core.agent import Agent
from agentos.core.tool import Tool


DelegationStatus = Literal["success", "failure", "timeout"]


@dataclass(frozen=True)
class DelegationRequest:
    """Structured handoff object for child agent execution.

    Design goal: avoid context compression loss by passing explicit fields
    (task/context/constraints) instead of a single free-form prose request.
    """

    task: str
    context_payload: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    expected_output_schema: dict[str, Any] | None = None
    timeout: float = 30.0


@dataclass(frozen=True)
class DelegationResponse:
    """Response returned from delegation manager/tool."""

    result: str
    status: DelegationStatus
    metadata: dict[str, Any] = field(default_factory=dict)


class DelegationManager:
    """Registry + router for agent-to-agent subtasks.

    The manager can be shared across multiple parent agents. It provides:
    - a registry of available child agents
    - a delegation method that executes a structured request
    - a tool factory that exposes delegation to LLMs
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._lock = threading.Lock()

        # Best-effort shared context store for future extensions.
        # Currently, delegation still includes `context_payload` in the prompt,
        # but we keep the payload in-memory under a key so an attached
        # "read_context" tool could be added later.
        self._context_store: dict[str, dict[str, Any]] = {}

        # Separate pool so parent agent threads don't block.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8
        )

    def register_agent(self, name: str, agent: Agent) -> None:
        with self._lock:
            self._agents[name] = agent

    def unregister_agent(self, name: str) -> None:
        with self._lock:
            self._agents.pop(name, None)

    def get_agent(self, name: str) -> Agent | None:
        with self._lock:
            return self._agents.get(name)

    def list_agents(self) -> list[str]:
        with self._lock:
            return list(self._agents.keys())

    def _choose_target_agent(
        self, *, constraints: dict[str, Any], explicit_name: str | None
    ) -> tuple[str, Agent]:
        if explicit_name:
            agent = self.get_agent(explicit_name)
            if agent is None:
                raise KeyError(f"Unknown child agent: {explicit_name}")
            return explicit_name, agent

        # Convention: allow target agent selection via constraints.
        for key in ("agent", "agent_name", "child_agent", "target_agent"):
            if key in constraints and isinstance(constraints[key], str):
                target = constraints[key]
                agent = self.get_agent(target)
                if agent is None:
                    raise KeyError(f"Unknown child agent: {target}")
                return target, agent

        with self._lock:
            if len(self._agents) == 1:
                only = next(iter(self._agents.keys()))
                return only, self._agents[only]

        raise ValueError(
            "No child agent specified. Provide `child_agent` in tool args "
            "or set one of constraints keys: 'agent', 'agent_name', 'child_agent', "
            "'target_agent'."
        )

    def _summarize_child_events(self, child: Agent) -> dict[str, Any]:
        # Agent stores events on `child.events`; we keep a compact summary.
        counts: dict[str, int] = {}
        tools_called: list[str] = []
        latencies: list[float] = []
        for e in getattr(child, "events", []) or []:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
            latencies.append(getattr(e, "latency_ms", 0.0) or 0.0)
            data = getattr(e, "data", {}) or {}
            if e.event_type == "tool_call" and isinstance(data, dict):
                tool_name = data.get("tool")
                if isinstance(tool_name, str):
                    tools_called.append(tool_name)

        lat_sum = float(sum(latencies)) if latencies else 0.0
        lat_max = float(max(latencies)) if latencies else 0.0

        return {
            "event_type_counts": counts,
            "tools_called": tools_called[-20:],
            "latency_ms_sum": round(lat_sum, 3),
            "latency_ms_max": round(lat_max, 3),
        }

    def delegate(
        self,
        request: DelegationRequest,
        *,
        child_agent_name: str | None = None,
    ) -> DelegationResponse:
        """Execute `request` on the chosen child agent with a timeout."""

        started = time.time()
        shared_context_key = uuid.uuid4().hex
        # Store context for potential shared-memory reading patterns.
        self._context_store[shared_context_key] = request.context_payload

        target_name, child = self._choose_target_agent(
            constraints=request.constraints, explicit_name=child_agent_name
        )

        # Provide a structured prompt including explicit fields.
        # This is intentionally JSON-ish rather than prose to keep fields
        # recoverable by the model (reducing context compression loss).
        prompt = {
            "delegation": {
                "task": request.task,
                "context": request.context_payload,
                "constraints": request.constraints,
                "expected_output_schema": request.expected_output_schema,
                "shared_context_key": shared_context_key,
            }
        }
        user_input = (
            "You have been delegated a structured subtask.\n"
            "Return ONLY the final answer content.\n"
            f"DelegationRequest (JSON): {json.dumps(prompt, ensure_ascii=False)}"
        )

        future = self._executor.submit(child.run, user_input)
        try:
            msg = future.result(timeout=request.timeout)
            result_text = (msg.content or "") if msg is not None else ""
            elapsed_ms = (time.time() - started) * 1000
            return DelegationResponse(
                result=result_text,
                status="success",
                metadata={
                    "child_agent": target_name,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "shared_context_key": shared_context_key,
                    "child_events_summary": self._summarize_child_events(child),
                },
            )
        except concurrent.futures.TimeoutError:
            elapsed_ms = (time.time() - started) * 1000
            return DelegationResponse(
                result="",
                status="timeout",
                metadata={
                    "child_agent": target_name,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "shared_context_key": shared_context_key,
                },
            )
        except Exception as e:
            elapsed_ms = (time.time() - started) * 1000
            return DelegationResponse(
                result=f"ERROR: {e}",
                status="failure",
                metadata={
                    "child_agent": target_name,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "shared_context_key": shared_context_key,
                    "error": str(e),
                },
            )

    def _delegate_tool_fn(
        self,
        child_agent_name: str,
        task: str,
        context_json: str,
        constraints_json: str,
        expected_output_schema_json: str,
        timeout: float,
    ) -> str:
        # Tool interface expects JSON strings for dict-like fields.
        def _parse_obj(s: str) -> dict[str, Any]:
            if not s:
                return {}
            try:
                v = json.loads(s)
            except json.JSONDecodeError:
                # Allow passing already-stringified primitives as best-effort.
                return {"value": s}
            if isinstance(v, dict):
                return v
            return {"value": v}

        context_payload = _parse_obj(context_json)
        constraints = _parse_obj(constraints_json)
        expected_output_schema = None
        if expected_output_schema_json:
            try:
                expected_output_schema = json.loads(expected_output_schema_json)
            except json.JSONDecodeError:
                expected_output_schema = {"value": expected_output_schema_json}

        req = DelegationRequest(
            task=task,
            context_payload=context_payload,
            constraints=constraints,
            expected_output_schema=expected_output_schema,
            timeout=float(timeout),
        )
        resp = self.delegate(req, child_agent_name=child_agent_name)
        return json.dumps(resp.__dict__, ensure_ascii=False)

    def build_delegate_tool(
        self,
        *,
        name: str = "delegate_subtask",
        description: str = "Delegate a structured subtask to another agent",
        timeout_seconds: float = 60.0,
    ) -> Tool:
        """Create a Tool that delegates subtasks via this manager."""

        def _tool_fn(
            child_agent_name: str,
            task: str,
            context_json: str,
            constraints_json: str,
            expected_output_schema_json: str,
            timeout: float = 30.0,
        ) -> str:
            return self._delegate_tool_fn(
                child_agent_name=child_agent_name,
                task=task,
                context_json=context_json,
                constraints_json=constraints_json,
                expected_output_schema_json=expected_output_schema_json,
                timeout=timeout,
            )

        return Tool(
            fn=_tool_fn,
            name=name,
            description=description,
            timeout_seconds=timeout_seconds,
        )

    def attach_delegate_tool(
        self,
        agent: Agent,
        *,
        tool_name: str = "delegate_subtask",
        tool_description: str = "Delegate a structured subtask to another agent",
    ) -> Tool:
        """Attach a delegation tool to `agent`'s toolset."""

        tool = self.build_delegate_tool(
            name=tool_name, description=tool_description
        )
        agent.tools.append(tool)
        agent._tool_map[tool.name] = tool
        return tool


__all__ = [
    "DelegationRequest",
    "DelegationResponse",
    "DelegationManager",
]


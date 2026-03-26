from __future__ import annotations

import concurrent.futures
import contextvars
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from agentos.core.agent import Agent
from agentos.core.tool import Tool


DelegationStatus = Literal["success", "failure", "timeout"]

_log = logging.getLogger("agentos.delegation")


class SharedContext:
    """In-memory key/value store shared across a delegation chain."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(values)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def dump(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


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

        self._shared_contexts: dict[str, SharedContext] = {}

        # contextvars allow delegation tools to discover the currently running
        # delegation chain (shared context + cancellation) without requiring
        # the LLM to pass the keys around.
        self._current_shared_context_key: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar(
                "agentos_delegation_shared_context_key", default=None
            )
        )
        self._current_cancel_event: contextvars.ContextVar[threading.Event | None] = (
            contextvars.ContextVar("agentos_delegation_cancel_event", default=None)
        )
        self._current_delegation_id: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar("agentos_delegation_id", default=None)
        )
        self._current_parent_delegation_id: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar("agentos_delegation_parent_id", default=None)
        )
        self._current_delegation_depth: contextvars.ContextVar[int] = (
            contextvars.ContextVar("agentos_delegation_depth", default=0)
        )

        # Separate pool so parent agent threads don't block.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

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

    def _get_or_create_shared_context(self, shared_context_key: str) -> SharedContext:
        with self._lock:
            sc = self._shared_contexts.get(shared_context_key)
            if sc is None:
                sc = SharedContext()
                self._shared_contexts[shared_context_key] = sc
            return sc

    def _can_continue(self) -> bool:
        ev = self._current_cancel_event.get()
        return ev is None or not ev.is_set()

    def _ensure_shared_context_tools_attached(self, agent: Agent) -> None:
        """Attach read/write tools for the active shared-context chain.

        The tools consult manager contextvars at runtime to select the
        appropriate shared_context_key for the currently executing chain.
        """

        # Attach only once per agent (by tool name).
        existing = getattr(agent, "_tool_map", {})

        if "shared_context_key" not in existing:

            def shared_context_key() -> str:
                key = self._current_shared_context_key.get()
                return key or ""

            tool = Tool(
                fn=shared_context_key,
                name="shared_context_key",
                description="Return the shared_context_key for the current delegation chain.",
                timeout_seconds=30.0,
            )
            agent.tools.append(tool)
            agent._tool_map[tool.name] = tool

        if "shared_context_get" not in existing:

            def shared_context_get(key: str) -> str:
                if not self._can_continue():
                    return json.dumps(
                        {"error": "delegation_cancelled", "key": key},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                sc_key = self._current_shared_context_key.get()
                if not sc_key:
                    return json.dumps(
                        {"error": "no_shared_context"}, ensure_ascii=False
                    )
                sc = self._get_or_create_shared_context(sc_key)
                return json.dumps(
                    {"key": key, "value": sc.get(key)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            tool = Tool(
                fn=shared_context_get,
                name="shared_context_get",
                description="Get a value from the current shared context (by key). Returns a JSON string {key,value}.",
                timeout_seconds=30.0,
            )
            agent.tools.append(tool)
            agent._tool_map[tool.name] = tool

        if "shared_context_set" not in existing:

            def shared_context_set(key: str, value_json: str) -> str:
                if not self._can_continue():
                    return json.dumps(
                        {"error": "delegation_cancelled", "key": key},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                sc_key = self._current_shared_context_key.get()
                if not sc_key:
                    return json.dumps(
                        {"error": "no_shared_context"}, ensure_ascii=False
                    )
                sc = self._get_or_create_shared_context(sc_key)
                try:
                    value = json.loads(value_json) if value_json else value_json
                except json.JSONDecodeError:
                    value = value_json
                sc.set(key, value)
                return json.dumps({"ok": True, "key": key}, ensure_ascii=False)

            tool = Tool(
                fn=shared_context_set,
                name="shared_context_set",
                description="Set a value in the current shared context. value_json should be JSON (or a raw string).",
                timeout_seconds=30.0,
            )
            agent.tools.append(tool)
            agent._tool_map[tool.name] = tool

        if "shared_context_dump" not in existing:

            def shared_context_dump() -> str:
                if not self._can_continue():
                    return json.dumps(
                        {"error": "delegation_cancelled"}, ensure_ascii=False
                    )
                sc_key = self._current_shared_context_key.get()
                if not sc_key:
                    return json.dumps(
                        {"error": "no_shared_context"}, ensure_ascii=False
                    )
                sc = self._get_or_create_shared_context(sc_key)
                return json.dumps(
                    {"shared_context_key": sc_key, "data": sc.dump()},
                    ensure_ascii=False,
                )

            tool = Tool(
                fn=shared_context_dump,
                name="shared_context_dump",
                description="Dump the entire current shared context as JSON.",
                timeout_seconds=30.0,
            )
            agent.tools.append(tool)
            agent._tool_map[tool.name] = tool

    def delegate(
        self,
        request: DelegationRequest,
        *,
        child_agent_name: str | None = None,
    ) -> DelegationResponse:
        """Execute `request` on the chosen child agent with a timeout.

        Delegation chains reuse the same shared context automatically via
        contextvars, enabling child agents to read/write rich context without
        embedding the entire payload in the prompt.
        """

        started = time.time()
        parent_delegation_id = self._current_delegation_id.get()
        delegation_id = uuid.uuid4().hex
        depth = self._current_delegation_depth.get() or 0
        indent = "  " * depth

        inherited_shared_context_key = self._current_shared_context_key.get()
        shared_context_key = inherited_shared_context_key or uuid.uuid4().hex

        cancel_event = threading.Event()

        target_name, child = self._choose_target_agent(
            constraints=request.constraints, explicit_name=child_agent_name
        )

        # Write structured context payload into the shared store.
        sc = self._get_or_create_shared_context(shared_context_key)
        if request.context_payload:
            sc.update(request.context_payload)
            _log.info(
                "%s  context_pass shared_context_key=%s keys=%s",
                indent,
                shared_context_key,
                list(request.context_payload.keys())[:50],
            )

        prompt = {
            "delegation": {
                "task": request.task,
                "context": {
                    "shared_context_key": shared_context_key,
                    "context_payload_keys": list(request.context_payload.keys()),
                },
                "constraints": request.constraints,
                "expected_output_schema": request.expected_output_schema,
                "delegation_id": delegation_id,
                "parent_delegation_id": parent_delegation_id,
            }
        }

        user_input = (
            "You have been delegated a structured subtask.\n"
            "Return ONLY the final answer content.\n"
            "Use the shared_context_* tools to read/write rich state.\n"
            f"DelegationRequest (JSON): {json.dumps(prompt, ensure_ascii=False, separators=(',', ':'))}"
        )

        _log.info(
            "%sSTART delegation id=%s parent=%s child_agent=%s shared_context_key=%s timeout=%ss task=%r",
            indent,
            delegation_id,
            parent_delegation_id,
            target_name,
            shared_context_key,
            request.timeout,
            request.task[:200] if request.task else "",
        )

        # Attach shared context tools so the delegated agent can read/write.
        self._ensure_shared_context_tools_attached(child)

        # Ensure it can chain further delegation.
        # If delegation tool not present, attach a default one.
        if "delegate_subtask" not in getattr(child, "_tool_map", {}):
            self.attach_delegate_tool(child)

        try:
            t_shared = self._current_shared_context_key.set(shared_context_key)
            t_cancel = self._current_cancel_event.set(cancel_event)
            t_id = self._current_delegation_id.set(delegation_id)
            t_parent = self._current_parent_delegation_id.set(parent_delegation_id)
            t_depth = self._current_delegation_depth.set(depth + 1)

            # Capture contextvars AFTER setting tokens so the delegated
            # `child.run()` thread sees the correct delegation chain state.
            ctx = contextvars.copy_context()

            future = self._executor.submit(
                ctx.run,
                child.run,
                user_input,
            )
            token_reset = (t_shared, t_cancel, t_id, t_parent, t_depth)

            try:
                msg = future.result(timeout=request.timeout)
                result_text = (msg.content or "") if msg is not None else ""
                elapsed_ms = (time.time() - started) * 1000
                _log.info(
                    "%sEND delegation id=%s status=success elapsed_ms=%.1f child_agent=%s",
                    indent,
                    delegation_id,
                    elapsed_ms,
                    target_name,
                )
                return DelegationResponse(
                    result=result_text,
                    status="success",
                    metadata={
                        "child_agent": target_name,
                        "delegation_id": delegation_id,
                        "parent_delegation_id": parent_delegation_id,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "shared_context_key": shared_context_key,
                        "child_events_summary": self._summarize_child_events(child),
                    },
                )
            except concurrent.futures.TimeoutError:
                elapsed_ms = (time.time() - started) * 1000
                cancel_event.set()
                future.cancel()
                _log.warning(
                    "%sTIMEOUT delegation id=%s child_agent=%s elapsed_ms=%.1f shared_context_key=%s",
                    indent,
                    delegation_id,
                    target_name,
                    elapsed_ms,
                    shared_context_key,
                )
                return DelegationResponse(
                    result="",
                    status="timeout",
                    metadata={
                        "child_agent": target_name,
                        "delegation_id": delegation_id,
                        "parent_delegation_id": parent_delegation_id,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "shared_context_key": shared_context_key,
                    },
                )
            except Exception as e:
                elapsed_ms = (time.time() - started) * 1000
                cancel_event.set()
                _log.exception(
                    "%sFAIL delegation id=%s child_agent=%s elapsed_ms=%.1f err=%s",
                    indent,
                    delegation_id,
                    target_name,
                    elapsed_ms,
                    str(e),
                )
                return DelegationResponse(
                    result=f"ERROR: {e}",
                    status="failure",
                    metadata={
                        "child_agent": target_name,
                        "delegation_id": delegation_id,
                        "parent_delegation_id": parent_delegation_id,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "shared_context_key": shared_context_key,
                        "error": str(e),
                    },
                )
        finally:
            # Reset contextvars for this thread.
            try:
                if "token_reset" in locals():  # noqa: SIM102
                    t_shared, t_cancel, t_id, t_parent, t_depth = token_reset
                    self._current_shared_context_key.reset(t_shared)
                    self._current_cancel_event.reset(t_cancel)
                    self._current_delegation_id.reset(t_id)
                    self._current_parent_delegation_id.reset(t_parent)
                    self._current_delegation_depth.reset(t_depth)
            except Exception:
                pass

    def _delegate_tool_fn(
        self,
        child_agent_name: str,
        task: str,
        context_json: str,
        constraints_json: str,
        expected_output_schema_json: str,
        timeout: float,
    ) -> str:
        # If the current delegation chain timed out/cancelled, avoid starting
        # new nested work.
        if not self._can_continue():
            cancelled = DelegationResponse(
                result="",
                status="timeout",
                metadata={
                    "child_agent": child_agent_name,
                    "error": "delegation_cancelled",
                },
            )
            return json.dumps(cancelled.__dict__, ensure_ascii=False)

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

        tool = self.build_delegate_tool(name=tool_name, description=tool_description)
        agent.tools.append(tool)
        agent._tool_map[tool.name] = tool

        # Make shared-context read/write available to this agent so it can
        # participate in delegation chains (and grandchildren can chain too).
        self._ensure_shared_context_tools_attached(agent)
        return tool


__all__ = [
    "SharedContext",
    "DelegationRequest",
    "DelegationResponse",
    "DelegationManager",
]

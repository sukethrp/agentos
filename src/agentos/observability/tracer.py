"""Deep Tracer — capture every decision an agent makes.

Records a structured :class:`Trace` for each ``agent.run()`` call.
Each trace contains a sequence of :class:`TraceStep` objects that
capture exactly:

* What the LLM **saw** (full message list, system prompt, tool schemas)
* What it **decided** (tool call, final answer, or error)
* **Why** (the reasoning visible in the response and tool outputs)
* Timing, cost, and token counts per step

Traces can be replayed, compared, or fed into the diagnostics engine.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class TraceStep:
    """One atomic step inside a trace."""

    step_index: int = 0
    step_type: StepType = StepType.LLM_CALL
    timestamp: float = 0.0

    # What the LLM saw
    messages_snapshot: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    available_tools: list[str] = field(default_factory=list)

    # What it decided
    decision: str = ""               # "call tool X" | "final answer" | "error"
    tool_name: str = ""
    tool_arguments: dict = field(default_factory=dict)
    tool_result: str = ""
    response_text: str = ""

    # Metrics
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    # Diagnostics hooks
    is_error: bool = False
    error_message: str = ""
    tool_not_found: bool = False

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "step_type": self.step_type.value,
            "timestamp": self.timestamp,
            "system_prompt": self.system_prompt[:200] if self.system_prompt else "",
            "message_count": len(self.messages_snapshot),
            "available_tools": self.available_tools,
            "decision": self.decision,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result[:300] if self.tool_result else "",
            "response_text": self.response_text[:300] if self.response_text else "",
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "is_error": self.is_error,
            "error_message": self.error_message,
            "tool_not_found": self.tool_not_found,
        }

    def summary(self) -> str:
        if self.step_type == StepType.TOOL_CALL:
            return f"[{self.step_index}] TOOL {self.tool_name}({self.tool_arguments}) → {self.tool_result[:60]}"
        if self.step_type == StepType.FINAL_ANSWER:
            return f"[{self.step_index}] ANSWER: {self.response_text[:80]}"
        if self.step_type == StepType.ERROR:
            return f"[{self.step_index}] ERROR: {self.error_message[:80]}"
        return f"[{self.step_index}] LLM_CALL ({self.tokens_used} tokens, {self.latency_ms:.0f}ms)"


@dataclass
class Trace:
    """Complete trace of one agent.run() invocation."""

    trace_id: str = ""
    agent_name: str = ""
    model: str = ""
    user_query: str = ""
    final_response: str = ""
    system_prompt: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    steps: list[TraceStep] = field(default_factory=list)
    success: bool = True
    error: str = ""

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = uuid.uuid4().hex[:12]

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.steps)

    @property
    def llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.step_type == StepType.LLM_CALL)

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.step_type == StepType.TOOL_CALL)

    @property
    def errors(self) -> list[TraceStep]:
        return [s for s in self.steps if s.is_error]

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at) * 1000
        return self.total_latency_ms

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "user_query": self.user_query,
            "final_response": self.final_response[:500],
            "system_prompt": self.system_prompt[:300],
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 1),
            "total_cost": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "step_count": len(self.steps),
            "success": self.success,
            "error": self.error,
            "steps": [s.to_dict() for s in self.steps],
        }

    def timeline(self) -> str:
        """Human-readable timeline."""
        lines = [
            f"Trace {self.trace_id}  [{self.agent_name}]",
            f"Query: {self.user_query[:80]}",
            f"Model: {self.model}  Duration: {self.duration_ms:.0f}ms  Cost: ${self.total_cost:.4f}",
            "-" * 60,
        ]
        for s in self.steps:
            lines.append(f"  {s.summary()}")
        lines.append("-" * 60)
        status = "✅ Success" if self.success else f"❌ Failed: {self.error}"
        lines.append(f"  {status}")
        return "\n".join(lines)


# ── Trace builder (used by the agent or a wrapper) ──────────────────────────

class TraceBuilder:
    """Incrementally build a Trace during an agent run."""

    def __init__(self, agent_name: str = "", model: str = "", system_prompt: str = "") -> None:
        self._trace = Trace(
            agent_name=agent_name,
            model=model,
            system_prompt=system_prompt,
            started_at=time.time(),
        )
        self._step_counter = 0

    def set_query(self, query: str) -> None:
        self._trace.user_query = query

    def add_llm_call(
        self,
        messages: list[dict],
        available_tools: list[str],
        tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
    ) -> TraceStep:
        step = TraceStep(
            step_index=self._step_counter,
            step_type=StepType.LLM_CALL,
            timestamp=time.time(),
            messages_snapshot=copy.deepcopy(messages),
            system_prompt=self._trace.system_prompt,
            available_tools=available_tools,
            decision="llm_call",
            tokens_used=tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        self._trace.steps.append(step)
        self._step_counter += 1
        return step

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: str = "",
        latency_ms: float = 0.0,
        not_found: bool = False,
    ) -> TraceStep:
        is_err = not_found or result.startswith("ERROR")
        step = TraceStep(
            step_index=self._step_counter,
            step_type=StepType.TOOL_CALL,
            timestamp=time.time(),
            tool_name=tool_name,
            tool_arguments=arguments,
            tool_result=result,
            decision=f"call {tool_name}",
            latency_ms=latency_ms,
            is_error=is_err,
            error_message=result if is_err else "",
            tool_not_found=not_found,
        )
        self._trace.steps.append(step)
        self._step_counter += 1
        return step

    def add_final_answer(self, response: str) -> TraceStep:
        step = TraceStep(
            step_index=self._step_counter,
            step_type=StepType.FINAL_ANSWER,
            timestamp=time.time(),
            response_text=response,
            decision="final_answer",
        )
        self._trace.steps.append(step)
        self._trace.final_response = response
        self._step_counter += 1
        return step

    def add_error(self, message: str) -> TraceStep:
        step = TraceStep(
            step_index=self._step_counter,
            step_type=StepType.ERROR,
            timestamp=time.time(),
            decision="error",
            is_error=True,
            error_message=message,
        )
        self._trace.steps.append(step)
        self._trace.success = False
        self._trace.error = message
        self._step_counter += 1
        return step

    def finish(self) -> Trace:
        self._trace.ended_at = time.time()
        return self._trace


# ── Trace store ──────────────────────────────────────────────────────────────

class TraceStore:
    """In-memory store for traces."""

    def __init__(self, max_traces: int = 500) -> None:
        self._traces: list[Trace] = []
        self.max_traces = max_traces

    def add(self, trace: Trace) -> None:
        self._traces.append(trace)
        if len(self._traces) > self.max_traces:
            self._traces = self._traces[-self.max_traces:]

    def get(self, trace_id: str) -> Trace | None:
        for t in reversed(self._traces):
            if t.trace_id == trace_id:
                return t
        return None

    def list_all(self, agent_name: str = "", limit: int = 50) -> list[Trace]:
        result = self._traces
        if agent_name:
            result = [t for t in result if t.agent_name == agent_name]
        return sorted(result, key=lambda t: -t.started_at)[:limit]

    def failed(self, limit: int = 20) -> list[Trace]:
        return [t for t in sorted(self._traces, key=lambda t: -t.started_at) if not t.success][:limit]

    def stats(self) -> dict:
        total = len(self._traces)
        failed = sum(1 for t in self._traces if not t.success)
        return {
            "total_traces": total,
            "failed": failed,
            "success_rate": round((total - failed) / max(total, 1) * 100, 1),
            "agents": sorted({t.agent_name for t in self._traces}),
        }


_default_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _default_store
    if _default_store is None:
        _default_store = TraceStore()
    return _default_store

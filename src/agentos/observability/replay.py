"""Replay — step through a failed interaction to see exactly what went wrong.

Given a :class:`Trace`, produce a human-readable (and machine-parseable)
step-by-step replay that shows:

* The full message context at each LLM call
* The tool selected and why
* The tool's return value
* The LLM's interpretation
* Where the chain broke

Can be printed to the console or serialised for the web UI.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

from agentos.observability.tracer import Trace, TraceStep, StepType
from agentos.observability.diagnostics import Diagnosis, diagnose, Severity


@dataclass
class ReplayFrame:
    """One frame in the step-by-step replay."""

    frame_index: int = 0
    label: str = ""
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    is_failure_point: bool = False
    severity: str = "ok"           # "ok" | "warn" | "fail"

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "label": self.label,
            "detail": self.detail,
            "data": self.data,
            "is_failure_point": self.is_failure_point,
            "severity": self.severity,
        }


@dataclass
class Replay:
    """Complete step-by-step replay of an interaction."""

    trace_id: str = ""
    agent_name: str = ""
    user_query: str = ""
    frames: list[ReplayFrame] = field(default_factory=list)
    diagnosis: Diagnosis | None = None
    failure_frame: int | None = None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "user_query": self.user_query[:120],
            "frame_count": len(self.frames),
            "failure_frame": self.failure_frame,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "frames": [f.to_dict() for f in self.frames],
        }

    def text(self) -> str:
        """Pretty-print the replay for console output."""
        lines = [
            "=" * 70,
            f"  REPLAY — Trace {self.trace_id}  [{self.agent_name}]",
            "=" * 70,
            f"  Query: {self.user_query}",
            "",
        ]

        for f in self.frames:
            icon = "❌" if f.is_failure_point else ("⚠️" if f.severity == "warn" else "▶")
            pointer = " ← FAILURE POINT" if f.is_failure_point else ""
            lines.append(f"  {icon} Frame {f.frame_index}: {f.label}{pointer}")
            if f.detail:
                for line in f.detail.split("\n"):
                    lines.append(f"     {line}")
            lines.append("")

        if self.diagnosis:
            lines.append("  ─── Diagnosis ───")
            lines.append(f"  Root cause: {self.diagnosis.root_cause}")
            for c in self.diagnosis.checks:
                sev_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(c.severity.value, "?")
                lines.append(f"    {sev_icon} {c.check_name}: {c.title}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ── Replay builder ───────────────────────────────────────────────────────────

def build_replay(trace: Trace, include_messages: bool = False) -> Replay:
    """Build a step-by-step replay from a trace with full diagnosis."""
    diag = diagnose(trace)

    replay = Replay(
        trace_id=trace.trace_id,
        agent_name=trace.agent_name,
        user_query=trace.user_query,
        diagnosis=diag,
    )

    frame_idx = 0

    # Frame 0: Setup
    setup_detail = (
        f"Agent: {trace.agent_name}\n"
        f"Model: {trace.model}\n"
        f"System prompt: {trace.system_prompt[:120]}{'…' if len(trace.system_prompt) > 120 else ''}"
    )
    replay.frames.append(ReplayFrame(
        frame_index=frame_idx,
        label="SETUP — Agent initialised",
        detail=setup_detail,
        data={"model": trace.model, "system_prompt_length": len(trace.system_prompt)},
    ))
    frame_idx += 1

    # Frame 1: User query
    replay.frames.append(ReplayFrame(
        frame_index=frame_idx,
        label="USER QUERY",
        detail=trace.user_query,
    ))
    frame_idx += 1

    # Build frames for each step
    for step in trace.steps:
        frame = _step_to_frame(step, frame_idx, diag, include_messages)
        replay.frames.append(frame)
        if frame.is_failure_point and replay.failure_frame is None:
            replay.failure_frame = frame_idx
        frame_idx += 1

    # Final outcome frame
    if trace.success:
        replay.frames.append(ReplayFrame(
            frame_index=frame_idx,
            label="OUTCOME — Success",
            detail=f"Final response ({len(trace.final_response)} chars): {trace.final_response[:200]}",
            severity="ok",
        ))
    else:
        replay.frames.append(ReplayFrame(
            frame_index=frame_idx,
            label="OUTCOME — Failed",
            detail=trace.error or "Agent did not produce a successful response",
            severity="fail",
            is_failure_point=replay.failure_frame is None,
        ))
        if replay.failure_frame is None:
            replay.failure_frame = frame_idx

    return replay


def _step_to_frame(step: TraceStep, frame_idx: int, diag: Diagnosis, include_messages: bool) -> ReplayFrame:
    """Convert a TraceStep into a ReplayFrame."""

    is_failure = step.is_error
    # Also mark as failure if the diagnosis points to this step
    if diag.root_cause_step == step.step_index and diag.overall_severity == Severity.FAIL:
        is_failure = True

    severity = "fail" if is_failure else ("warn" if step.is_error else "ok")

    if step.step_type == StepType.LLM_CALL:
        detail_lines = [
            f"Messages: {len(step.messages_snapshot)} in context",
            f"Available tools: {', '.join(step.available_tools) or 'none'}",
            f"Tokens: {step.tokens_used}  Cost: ${step.cost_usd:.4f}  Latency: {step.latency_ms:.0f}ms",
        ]
        if include_messages and step.messages_snapshot:
            detail_lines.append("\nMessage preview:")
            for m in step.messages_snapshot[-3:]:
                role = m.get("role", "?")
                content = (m.get("content") or "")[:80]
                detail_lines.append(f"  [{role}] {content}")

        return ReplayFrame(
            frame_index=frame_idx,
            label=f"LLM CALL (step {step.step_index})",
            detail="\n".join(detail_lines),
            data={"tokens": step.tokens_used, "cost": step.cost_usd, "latency_ms": step.latency_ms},
            severity=severity,
            is_failure_point=is_failure,
        )

    elif step.step_type == StepType.TOOL_CALL:
        detail_lines = [
            f"Tool: {step.tool_name}",
            f"Arguments: {step.tool_arguments}",
            f"Result: {step.tool_result[:200]}",
        ]
        if step.tool_not_found:
            detail_lines.append("⚠️ Tool NOT FOUND — LLM hallucinated this tool name")
        if step.is_error:
            detail_lines.append(f"❌ Error: {step.error_message[:200]}")

        return ReplayFrame(
            frame_index=frame_idx,
            label=f"TOOL CALL — {step.tool_name} (step {step.step_index})",
            detail="\n".join(detail_lines),
            data={"tool": step.tool_name, "args": step.tool_arguments},
            severity=severity,
            is_failure_point=is_failure,
        )

    elif step.step_type == StepType.FINAL_ANSWER:
        return ReplayFrame(
            frame_index=frame_idx,
            label=f"FINAL ANSWER (step {step.step_index})",
            detail=step.response_text[:300],
            severity=severity,
            is_failure_point=is_failure,
        )

    elif step.step_type == StepType.ERROR:
        return ReplayFrame(
            frame_index=frame_idx,
            label=f"ERROR (step {step.step_index})",
            detail=step.error_message,
            severity="fail",
            is_failure_point=True,
        )

    return ReplayFrame(
        frame_index=frame_idx,
        label=f"STEP {step.step_index} ({step.step_type.value})",
        detail=step.decision,
        severity=severity,
    )

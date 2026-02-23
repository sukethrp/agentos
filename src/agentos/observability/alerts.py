"""Smart Alerts — not just "quality dropped" but WHY it dropped.

The alert engine correlates quality changes with observable causes:
- Tool errors spiking
- System prompt changes
- Model changes
- New failure patterns in specific topics
- Latency regressions

Each :class:`SmartAlert` names the probable *cause*, not just the symptom.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from agentos.observability.tracer import Trace, TraceStore, StepType, get_trace_store
from agentos.observability.diagnostics import Diagnosis, Severity, diagnose


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SmartAlert:
    """A causal alert that explains *why* something is wrong."""

    id: str = ""
    level: AlertLevel = AlertLevel.WARNING
    title: str = ""
    cause: str = ""  # e.g. "tool 'web_search' returning errors"
    impact: str = ""  # e.g. "quality dropped from 8.1 to 5.3"
    recommendation: str = ""  # e.g. "check web_search API key"
    agent_name: str = ""
    timestamp: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            import uuid

            self.id = uuid.uuid4().hex[:10]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "level": self.level.value,
            "title": self.title,
            "cause": self.cause,
            "impact": self.impact,
            "recommendation": self.recommendation,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "evidence": self.evidence,
        }

    def summary(self) -> str:
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
            self.level.value, "?"
        )
        return f"{icon} [{self.level.value.upper()}] {self.title} — {self.cause}"


# ── Alert generators ─────────────────────────────────────────────────────────


def _check_tool_errors(traces: list[Trace]) -> list[SmartAlert]:
    """Detect when specific tools start failing."""
    alerts: list[SmartAlert] = []
    tool_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "errors": 0})

    for t in traces:
        for s in t.steps:
            if s.step_type == StepType.TOOL_CALL:
                tool_stats[s.tool_name]["total"] += 1
                if s.is_error:
                    tool_stats[s.tool_name]["errors"] += 1

    for tool_name, stats in tool_stats.items():
        if stats["total"] >= 3:
            error_rate = stats["errors"] / stats["total"] * 100
            if error_rate >= 50:
                alerts.append(
                    SmartAlert(
                        level=AlertLevel.CRITICAL,
                        title=f"Tool '{tool_name}' is failing",
                        cause=f"Tool '{tool_name}' returned errors in {error_rate:.0f}% of calls ({stats['errors']}/{stats['total']})",
                        impact="Agent responses may be incomplete or hallucinated when this tool is needed",
                        recommendation=f"Check the '{tool_name}' tool implementation, API keys, and external service status",
                        evidence=stats,
                    )
                )
            elif error_rate >= 20:
                alerts.append(
                    SmartAlert(
                        level=AlertLevel.WARNING,
                        title=f"Tool '{tool_name}' error rate elevated",
                        cause=f"Tool '{tool_name}' has a {error_rate:.0f}% error rate",
                        impact="Some queries relying on this tool may fail",
                        recommendation=f"Monitor '{tool_name}' — may be intermittent API issue",
                        evidence=stats,
                    )
                )
    return alerts


def _check_quality_drop(
    traces: list[Trace], diagnoses: list[Diagnosis]
) -> list[SmartAlert]:
    """Detect quality drops and correlate with root causes."""
    alerts: list[SmartAlert] = []
    if len(traces) < 6:
        return alerts

    sorted_traces = sorted(traces, key=lambda t: t.started_at)
    mid = len(sorted_traces) // 2
    first_half = sorted_traces[:mid]
    second_half = sorted_traces[mid:]

    first_fail = sum(1 for t in first_half if not t.success) / max(len(first_half), 1)
    second_fail = sum(1 for t in second_half if not t.success) / max(
        len(second_half), 1
    )

    if second_fail > first_fail + 0.15 and second_fail > 0.2:
        # Quality dropped — find the cause from diagnoses
        recent_diag = [d for d in diagnoses if d.overall_severity == Severity.FAIL]
        cause_counts: Counter = Counter()
        for d in recent_diag:
            for c in d.checks:
                if c.severity == Severity.FAIL:
                    cause_counts[c.check_name] += 1

        top_cause = cause_counts.most_common(1)[0] if cause_counts else ("unknown", 0)
        cause_map = {
            "context_quality": "system prompt or context issues",
            "tool_selection": "the LLM is choosing wrong tools or hallucinating tool names",
            "tool_execution": "tool execution errors are corrupting agent responses",
            "interpretation": "the LLM is misinterpreting tool results",
            "faithfulness": "the LLM is not using available data in its answers",
        }

        alerts.append(
            SmartAlert(
                level=AlertLevel.CRITICAL,
                title="Quality regression detected",
                cause=f"Root cause: {cause_map.get(top_cause[0], top_cause[0])} ({top_cause[1]} occurrences)",
                impact=f"Failure rate rose from {first_fail * 100:.0f}% to {second_fail * 100:.0f}%",
                recommendation=f"Focus on fixing '{top_cause[0]}' — it's the most common failure point",
                evidence={
                    "first_half_fail_rate": round(first_fail * 100, 1),
                    "second_half_fail_rate": round(second_fail * 100, 1),
                    "top_cause": top_cause[0],
                    "cause_count": top_cause[1],
                },
            )
        )
    return alerts


def _check_latency_regression(traces: list[Trace]) -> list[SmartAlert]:
    """Detect latency spikes."""
    alerts: list[SmartAlert] = []
    if len(traces) < 6:
        return alerts

    sorted_traces = sorted(traces, key=lambda t: t.started_at)
    mid = len(sorted_traces) // 2
    first_latencies = [t.duration_ms for t in sorted_traces[:mid] if t.duration_ms > 0]
    second_latencies = [t.duration_ms for t in sorted_traces[mid:] if t.duration_ms > 0]

    if not first_latencies or not second_latencies:
        return alerts

    first_avg = statistics.mean(first_latencies)
    second_avg = statistics.mean(second_latencies)

    if second_avg > first_avg * 2 and second_avg > 5000:
        # Find which step type got slower
        step_latencies: dict[str, list[float]] = defaultdict(list)
        for t in sorted_traces[mid:]:
            for s in t.steps:
                if s.latency_ms > 0:
                    step_latencies[s.step_type.value].append(s.latency_ms)

        slowest = (
            max(step_latencies, key=lambda k: statistics.mean(step_latencies[k]))
            if step_latencies
            else "unknown"
        )

        alerts.append(
            SmartAlert(
                level=AlertLevel.WARNING,
                title="Latency regression",
                cause=f"'{slowest}' steps are the bottleneck",
                impact=f"Average response time rose from {first_avg:.0f}ms to {second_avg:.0f}ms",
                recommendation=f"Investigate '{slowest}' latency — may be model or tool API degradation",
                evidence={
                    "first_avg_ms": round(first_avg, 1),
                    "second_avg_ms": round(second_avg, 1),
                    "slowest_step": slowest,
                },
            )
        )
    return alerts


def _check_missing_tools(traces: list[Trace]) -> list[SmartAlert]:
    """Detect when the LLM keeps trying to call tools that don't exist."""
    alerts: list[SmartAlert] = []
    missing: Counter = Counter()
    for t in traces:
        for s in t.steps:
            if s.tool_not_found:
                missing[s.tool_name] += 1

    for tool_name, count in missing.most_common(3):
        if count >= 2:
            alerts.append(
                SmartAlert(
                    level=AlertLevel.WARNING,
                    title=f"LLM keeps hallucinating tool '{tool_name}'",
                    cause=f"Tool '{tool_name}' was called {count} times but doesn't exist",
                    impact="These interactions fail because the tool result is an error",
                    recommendation=f"Either add '{tool_name}' as a real tool or update the system prompt to clarify available tools",
                    evidence={"tool_name": tool_name, "attempts": count},
                )
            )
    return alerts


# ── Alert engine ─────────────────────────────────────────────────────────────


class AlertEngine:
    """Run all alert generators against recent traces."""

    def __init__(self, store: TraceStore | None = None) -> None:
        self.store = store or get_trace_store()
        self._alerts: list[SmartAlert] = []
        self._callbacks: list[Callable[[SmartAlert], None]] = []

    def on_alert(self, callback: Callable[[SmartAlert], None]) -> None:
        """Register a callback that fires when a new alert is generated."""
        self._callbacks.append(callback)

    def evaluate(self, agent_name: str = "", limit: int = 100) -> list[SmartAlert]:
        """Evaluate recent traces and generate alerts."""
        traces = self.store.list_all(agent_name=agent_name, limit=limit)
        diagnoses = [diagnose(t) for t in traces]

        new_alerts: list[SmartAlert] = []
        new_alerts.extend(_check_tool_errors(traces))
        new_alerts.extend(_check_quality_drop(traces, diagnoses))
        new_alerts.extend(_check_latency_regression(traces))
        new_alerts.extend(_check_missing_tools(traces))

        # Tag alerts with agent name
        for a in new_alerts:
            if not a.agent_name and agent_name:
                a.agent_name = agent_name

        # Fire callbacks
        for a in new_alerts:
            for cb in self._callbacks:
                try:
                    cb(a)
                except Exception:
                    pass

        self._alerts = new_alerts
        return new_alerts

    def get_alerts(self) -> list[SmartAlert]:
        return list(self._alerts)

    def critical(self) -> list[SmartAlert]:
        return [a for a in self._alerts if a.level == AlertLevel.CRITICAL]

"""AgentOS Observability — deep tracing, root cause analysis, and smart alerts.

Trace every decision an agent makes, diagnose failures at the exact step
that went wrong, generate causal alerts, and replay interactions
frame-by-frame.
"""

from agentos.observability.tracer import (
    StepType,
    Trace,
    TraceBuilder,
    TraceStep,
    TraceStore,
    get_trace_store,
)
from agentos.observability.diagnostics import (
    CheckResult,
    Diagnosis,
    Severity,
    diagnose,
    diagnose_batch,
)
from agentos.observability.alerts import (
    AlertEngine,
    AlertLevel,
    SmartAlert,
)
from agentos.observability.replay import (
    Replay,
    ReplayFrame,
    build_replay,
)

__all__ = [
    "StepType",
    "Trace",
    "TraceBuilder",
    "TraceStep",
    "TraceStore",
    "get_trace_store",
    "CheckResult",
    "Diagnosis",
    "Severity",
    "diagnose",
    "diagnose_batch",
    "AlertEngine",
    "AlertLevel",
    "SmartAlert",
    "Replay",
    "ReplayFrame",
    "build_replay",
]

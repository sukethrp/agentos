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
    # Tracer
    "StepType",
    "Trace",
    "TraceBuilder",
    "TraceStep",
    "TraceStore",
    "get_trace_store",
    # Diagnostics
    "CheckResult",
    "Diagnosis",
    "Severity",
    "diagnose",
    "diagnose_batch",
    # Alerts
    "AlertEngine",
    "AlertLevel",
    "SmartAlert",
    # Replay
    "Replay",
    "ReplayFrame",
    "build_replay",
]

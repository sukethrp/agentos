"""AgentOS Observability — deep tracing, root cause analysis, and smart alerts.

Trace every decision an agent makes, diagnose failures at the exact step
that went wrong, generate causal alerts, and view past interactions
frame-by-frame.

This package renders recorded runs for humans. Hermetic record/replay, which
re-executes runs for machines, lives in :mod:`agentos.replay`.
"""

from agentos.observability.alerts import (
    AlertEngine,
    AlertLevel,
    SmartAlert,
)
from agentos.observability.diagnostics import (
    CheckResult,
    Diagnosis,
    Severity,
    diagnose,
    diagnose_batch,
)
from agentos.observability.run_viewer import (
    RunView,
    ViewFrame,
    build_run_view,
)
from agentos.observability.tracer import (
    StepType,
    Trace,
    TraceBuilder,
    TraceStep,
    TraceStore,
    get_trace_store,
)

__all__ = [
    "AlertEngine",
    "AlertLevel",
    "CheckResult",
    "Diagnosis",
    "RunView",
    "Severity",
    "SmartAlert",
    "StepType",
    "Trace",
    "TraceBuilder",
    "TraceStep",
    "TraceStore",
    "ViewFrame",
    "build_run_view",
    "diagnose",
    "diagnose_batch",
    "get_trace_store",
]

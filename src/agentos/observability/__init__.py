"""AgentOS Observability — deep tracing, root cause analysis, and smart alerts.

Trace every decision an agent makes, diagnose failures at the exact step
that went wrong, generate causal alerts, and view past interactions
frame-by-frame.

This package renders recorded runs for humans. Hermetic record/replay, which
re-executes runs for machines, lives in :mod:`agentos.replay`.
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
from agentos.observability.run_viewer import (
    RunView,
    ViewFrame,
    build_run_view,
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
    "RunView",
    "ViewFrame",
    "build_run_view",
]

"""Deterministic record, replay, diff, and bisect for AgentOS.

Hermetic execution replay. Distinct from `agentos.observability`, which renders
recorded runs for humans; this module reproduces them for machines.

`agentos.replay.provider` is intentionally absent from these exports. It is the
only submodule here that imports `agentos.core`, and keeping it off the package
surface is what lets `import agentos.replay` stay stdlib-only. See ADR-008.
"""

from .diff import (
    Change,
    ChangeKind,
    DiffReport,
    Divergence,
    IncomparableError,
    compare_paths,
    compare_readers,
    diff_events,
    render_human,
)
from .schema import (
    SCHEMA_VERSION,
    EventStatus,
    RunHeader,
    SeamKind,
    TraceEvent,
    call_site_id,
    digest_obj,
    trace_digest,
)
from .seam import (
    DivergenceError,
    DivergencePolicy,
    Interceptor,
    NullInterceptor,
    Recorder,
    ReplayedError,
    Replayer,
    SeamCodecMismatch,
    current_interceptor,
    intercept,
    use_interceptor,
)
from .store import BlobStore, TraceReader, TraceWriter

__all__ = [
    "SCHEMA_VERSION",
    "BlobStore",
    "Change",
    "ChangeKind",
    "DiffReport",
    "Divergence",
    "DivergenceError",
    "DivergencePolicy",
    "EventStatus",
    "IncomparableError",
    "Interceptor",
    "NullInterceptor",
    "Recorder",
    "ReplayedError",
    "Replayer",
    "RunHeader",
    "SeamCodecMismatch",
    "SeamKind",
    "TraceEvent",
    "TraceReader",
    "TraceWriter",
    "call_site_id",
    "compare_paths",
    "compare_readers",
    "current_interceptor",
    "diff_events",
    "digest_obj",
    "intercept",
    "render_human",
    "trace_digest",
    "use_interceptor",
]

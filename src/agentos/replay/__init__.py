"""Deterministic record, replay, diff, and bisect for AgentOS.

Hermetic execution replay. Distinct from `agentos.observability`, which renders
recorded runs for humans; this module reproduces them for machines.

`agentos.replay.provider` is intentionally absent from these exports. It is the
only submodule here that imports `agentos.core`, and keeping it off the package
surface is what lets `import agentos.replay` stay stdlib-only. See ADR-008.
"""

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
    "DivergenceError",
    "DivergencePolicy",
    "EventStatus",
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
    "current_interceptor",
    "digest_obj",
    "intercept",
    "trace_digest",
    "use_interceptor",
]

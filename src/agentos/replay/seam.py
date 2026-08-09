"""Seam interception: the one place record and replay diverge.

Call sites never know whether they are being recorded, replayed, or ignored.
They wrap the nondeterministic operation in `intercept(...)` and hand over a
thunk. That is the whole contract:

    from agentos.replay.seam import intercept, call_site_id
    from agentos.replay.schema import SeamKind

    CS_COMPLETE = call_site_id(__name__, "Router.complete", SeamKind.PROVIDER)

    def complete(self, req):
        return intercept(SeamKind.PROVIDER, CS_COMPLETE, req.to_dict(),
                         lambda: self._provider.complete(req))

Replay keying is `(seam, call_site, ordinal)` where ordinal is the nth hit of
that call site in the run. The recorded `input_digest` is then compared as an
assertion, NOT used as part of the key. That distinction matters: keying on the
input would silently fall back to a live call whenever the prompt changed, and
we would lose the one signal we actually want, which is "the inputs to step 41
differ, so the real divergence happened at or before step 40."
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar

from .schema import (
    EventStatus,
    RunHeader,
    SeamKind,
    TraceEvent,
    call_site_id,
    digest_obj,
)
from .store import BlobStore, TraceWriter

T = TypeVar("T")

__all__ = [
    "DivergenceError",
    "DivergencePolicy",
    "Interceptor",
    "NullInterceptor",
    "Recorder",
    "ReplayedError",
    "Replayer",
    "SeamCodecMismatch",
    "call_site_id",
    "current_interceptor",
    "intercept",
    "use_interceptor",
]


class DivergencePolicy(str, Enum):
    STRICT = "strict"  # any mismatch raises. Default. Use in CI.
    LENIENT = "lenient"  # fall through to live call, mark run TAINTED
    RECORD_NEW = "record_new"  # M2: fork a new trace from the divergence point


class DivergenceError(RuntimeError):
    """Replay hit a call the recording does not account for."""

    def __init__(
        self,
        message: str,
        *,
        event: TraceEvent | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        super().__init__(message)
        self.event = event
        self.expected = expected
        self.actual = actual


class ReplayedError(RuntimeError):
    """Re-raised stand-in for an exception captured during recording."""

    def __init__(self, error_type: str, error_message: str) -> None:
        super().__init__(f"{error_type}: {error_message}")
        self.error_type = error_type
        self.error_message = error_message


class SeamCodecMismatch(RuntimeError):
    """The trace and this build disagree about what a seam's inputs mean.

    Raised at `Replayer` construction, deliberately before any event is
    compared. A codec fingerprint covers the set of digested field names plus
    the codec version, so a mismatch means every `input_digest` in the trace was
    computed over a different projection than the one this build produces.
    Reporting that as ordinary divergence would blame the agent for a change in
    the recording apparatus, which is the most expensive kind of wrong answer.
    """

    def __init__(
        self, seam: str, recorded: str | None, current: str | None, detail: str = ""
    ) -> None:
        super().__init__(
            f"seam codec mismatch for {seam!r}: trace recorded {recorded!r}, "
            f"this build produces {current!r}. "
            f"{detail}"
            "The digested field set or the codec version changed, so input "
            "digests from this trace are not comparable. Re-record the trace "
            "with the current build, or check out the build that recorded it."
        )
        self.seam = seam
        self.recorded = recorded
        self.current = current


class Interceptor(Protocol):
    def intercept(
        self,
        seam: SeamKind,
        call_site: str,
        input_obj: Any,
        thunk: Callable[[], T],
        *,
        name: str = "",
        agent_id: str = "root",
    ) -> T: ...


class NullInterceptor:
    """Tracing off. One attribute lookup and a direct call, no allocation."""

    def intercept(
        self,
        seam: SeamKind,
        call_site: str,
        input_obj: Any,
        thunk: Callable[[], T],
        *,
        name: str = "",
        agent_id: str = "root",
    ) -> T:
        return thunk()


_NULL = NullInterceptor()
_current: contextvars.ContextVar[Interceptor] = contextvars.ContextVar(
    "agentos_interceptor", default=_NULL
)


def current_interceptor() -> Interceptor:
    return _current.get()


@dataclass
class use_interceptor:
    """Context manager that installs an interceptor for the current context."""

    interceptor: Interceptor
    _token: Any = None

    def __enter__(self) -> Interceptor:
        self._token = _current.set(self.interceptor)
        return self.interceptor

    def __exit__(self, *exc: object) -> None:
        _current.reset(self._token)


def intercept(
    seam: SeamKind,
    call_site: str,
    input_obj: Any,
    thunk: Callable[[], T],
    *,
    name: str = "",
    agent_id: str = "root",
) -> T:
    return _current.get().intercept(
        seam, call_site, input_obj, thunk, name=name, agent_id=agent_id
    )


class _Counters:
    def __init__(self) -> None:
        self._n: dict[tuple[str, str], int] = {}
        self.lamport = 0

    def next_ordinal(self, seam: SeamKind, call_site: str) -> int:
        key = (seam.value, call_site)
        n = self._n.get(key, 0)
        self._n[key] = n + 1
        return n

    def tick(self) -> int:
        self.lamport += 1
        return self.lamport


class Recorder:
    """Runs the real thing and writes a trace."""

    def __init__(
        self, writer: TraceWriter, redactor: Callable[[Any], Any] | None = None
    ) -> None:
        self.writer = writer
        self.blobs: BlobStore = writer.blobs
        self.redactor = redactor or (lambda x: x)
        self._c = _Counters()
        self.events: list[TraceEvent] = []

    def intercept(
        self,
        seam: SeamKind,
        call_site: str,
        input_obj: Any,
        thunk: Callable[[], T],
        *,
        name: str = "",
        agent_id: str = "root",
    ) -> T:
        safe_input = self.redactor(input_obj)
        ordinal = self._c.next_ordinal(seam, call_site)
        ev = TraceEvent(
            event_id=uuid.uuid4().hex,
            run_id=self.writer_run_id,
            seq=self.writer.next_seq(),
            seam=seam,
            call_site=call_site,
            ordinal=ordinal,
            input_digest=digest_obj(safe_input),
            name=name,
            agent_id=agent_id,
            lamport=self._c.tick(),
            wall_start_ns=time.time_ns(),
        )
        try:
            result = thunk()
        except Exception as exc:
            ev.status = EventStatus.ERROR
            ev.error_type = type(exc).__name__
            ev.error_message = str(exc)[:2000]
            ev.wall_end_ns = time.time_ns()
            self._emit(ev)
            raise
        ev.output_ref = self.blobs.put_obj(self.redactor(result))
        ev.wall_end_ns = time.time_ns()
        self._emit(ev)
        return result

    @property
    def writer_run_id(self) -> str:
        return self.writer.path.stem

    def _emit(self, ev: TraceEvent) -> None:
        self.writer.append(ev)
        self.events.append(ev)


def _check_seam_codecs(recorded: Mapping[str, str], current: Mapping[str, str]) -> None:
    """Refuse to replay a trace whose seam codecs differ from this build's.

    Asymmetric on purpose:

    - a seam the trace declares but this build does not provide is fatal, since
      nothing here knows how to read those payloads;
    - a differing fingerprint is fatal, since every input digest was taken over
      a different projection;
    - a seam this build provides but the trace never declares is allowed. That
      is what a 0.1.0 trace looks like after the 0.2.0 field defaults in, and
      what a run that simply never hit the seam looks like. Unknown is not the
      same as mismatched, and refusing here would make every pre-0.2.0 trace
      unreadable for no evidence.
    """
    for seam, recorded_fp in recorded.items():
        current_fp = current.get(seam)
        if current_fp is None:
            raise SeamCodecMismatch(
                seam,
                recorded_fp,
                None,
                f"This build provides no codec for {seam!r} "
                f"(it provides: {sorted(current) or 'none'}). ",
            )
        if current_fp != recorded_fp:
            raise SeamCodecMismatch(seam, recorded_fp, current_fp)


class Replayer:
    """Serves recorded outputs. Makes zero live calls under STRICT."""

    def __init__(
        self,
        events: list[TraceEvent],
        blobs: BlobStore,
        policy: DivergencePolicy = DivergencePolicy.STRICT,
        redactor: Callable[[Any], Any] | None = None,
        header: RunHeader | None = None,
        codecs: Mapping[str, str] | None = None,
    ) -> None:
        if header is not None and codecs is not None:
            _check_seam_codecs(header.seam_codecs, codecs)
        self.policy = policy
        self.blobs = blobs
        self.header = header
        # Must be the SAME redactor the recorder used, or every input digest
        # mismatches and every replay looks like a divergence.
        self.redactor = redactor or (lambda x: x)
        self._index: dict[tuple[str, str, int], TraceEvent] = {
            (e.seam.value, e.call_site, e.ordinal): e for e in events
        }
        self._c = _Counters()
        self.tainted = False
        self.live_calls = 0
        self.consumed: list[TraceEvent] = []

    def intercept(
        self,
        seam: SeamKind,
        call_site: str,
        input_obj: Any,
        thunk: Callable[[], T],
        *,
        name: str = "",
        agent_id: str = "root",
    ) -> T:
        ordinal = self._c.next_ordinal(seam, call_site)
        self._c.tick()
        rec = self._index.get((seam.value, call_site, ordinal))
        actual = digest_obj(self.redactor(input_obj))

        if rec is None:
            return self._diverge(
                f"unrecorded call: {seam.value}/{call_site} ordinal={ordinal}. "
                f"Control flow reached a seam the recording never hit.",
                thunk,
                expected=None,
                actual=actual,
            )

        if actual != rec.input_digest:
            return self._diverge(
                f"input mismatch at {seam.value}/{call_site} ordinal={ordinal} "
                f"(seq={rec.seq}). Inputs differ here, so the root cause is at or "
                f"before seq={rec.seq - 1}. Run `agentos diff` to localize.",
                thunk,
                event=rec,
                expected=rec.input_digest,
                actual=actual,
            )

        self.consumed.append(rec)
        if rec.status is EventStatus.ERROR:
            raise ReplayedError(rec.error_type or "Exception", rec.error_message or "")
        return self.blobs.get_obj(rec.output_ref)  # type: ignore[arg-type]

    def _diverge(
        self,
        message: str,
        thunk: Callable[[], T],
        *,
        event: TraceEvent | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ) -> T:
        if self.policy is DivergencePolicy.STRICT:
            raise DivergenceError(
                message, event=event, expected=expected, actual=actual
            )
        if self.policy is DivergencePolicy.LENIENT:
            self.tainted = True
            self.live_calls += 1
            return thunk()
        raise NotImplementedError(
            "RECORD_NEW is M2: fork a TraceWriter at the divergence point and "
            "continue recording. Needs a writer handle on the Replayer."
        )

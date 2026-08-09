"""Trace schema v0 for AgentOS deterministic replay.

Design invariants (see docs/DETERMINISM.md):

1. Every field that participates in equivalence is hashed via `canonical_json`.
2. Wall-clock fields are RECORDED but never authoritative. They are excluded
   from `event_digest`, so a slow replay is still an equivalent replay.
3. Payloads never live inline. They live in the content-addressed blob store
   and events reference them by digest. Events stay small and greppable.
4. `schema_version` is written into every run header. Readers refuse to load a
   major version they do not understand rather than guessing.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "0.1.0"
DIGEST_ALGO = "b2b"  # blake2b-256; swap for "b3" once blake3 is a dependency
_NULL_DIGEST = f"{DIGEST_ALGO}:" + "0" * 64


class SeamKind(str, Enum):
    """Every source of nondeterminism gets exactly one seam kind."""

    PROVIDER = "provider"  # LLM completion / embedding calls
    TOOL = "tool"  # registered agent tools with side effects
    CLOCK = "clock"  # time.time, monotonic, datetime.now
    ENTROPY = "entropy"  # random, numpy.random, uuid4
    ENV = "env"  # os.environ reads, config lookups
    HTTP = "http"  # raw outbound requests not behind a tool
    FS = "fs"  # filesystem reads/writes
    SCHEDULER = "scheduler"  # asyncio task interleaving decisions


class EventStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TAINTED = "tainted"  # replay fell through to a live call (LENIENT policy)


def canonical_json(obj: Any) -> bytes:
    """Deterministic serialization. Sorted keys, no whitespace, UTF-8, no NaN."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, bytes):
        return {"__bytes__": obj.hex()}
    raise TypeError(f"non-canonicalizable type in trace payload: {type(obj)!r}")


def digest_bytes(data: bytes) -> str:
    return f"{DIGEST_ALGO}:{hashlib.blake2b(data, digest_size=32).hexdigest()}"


def digest_obj(obj: Any) -> str:
    return digest_bytes(canonical_json(obj))


def call_site_id(module: str, qualname: str, seam: SeamKind, label: str = "") -> str:
    """Stable identity for a call location.

    Deliberately NOT line-number based; line numbers churn on every refactor and
    would invalidate every trace in the corpus. Module plus qualname plus an
    optional author-supplied label is stable across formatting changes.
    """
    return digest_obj([module, qualname, seam.value, label])[:24]


@dataclass(frozen=True, slots=True)
class RunHeader:
    """First line of every trace file. Everything needed to reproduce the run."""

    run_id: str
    schema_version: str = SCHEMA_VERSION
    created_at_ns: int = field(default_factory=time.time_ns)
    git_sha: str | None = None
    git_dirty: bool = False
    agentos_version: str | None = None
    config_digest: str = _NULL_DIGEST
    env_digest: str = _NULL_DIGEST
    seed: int = 0
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    policy: str = "strict"
    redactor_version: str = "0"
    labels: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def new(**kwargs: Any) -> RunHeader:
        return RunHeader(run_id=uuid.uuid4().hex, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["record_type"] = "header"
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> RunHeader:
        d = {k: v for k, v in d.items() if k != "record_type"}
        major = str(d.get("schema_version", "0")).split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"trace schema major version {major} is not readable by "
                f"{SCHEMA_VERSION}; run `agentos trace migrate`"
            )
        return RunHeader(**d)


@dataclass(slots=True)
class TraceEvent:
    """One interception at one seam.

    `seq` is the global record order. `lamport` plus `agent_id` carry causal
    order for concurrent agents; replay honors lamport, not wall clock.
    """

    event_id: str
    run_id: str
    seq: int
    seam: SeamKind
    call_site: str
    ordinal: int  # nth hit of this call_site in this run
    input_digest: str
    name: str = ""
    parent_id: str | None = None
    agent_id: str = "root"
    lamport: int = 0
    output_ref: str | None = None  # blob digest, None on error
    status: EventStatus = EventStatus.OK
    error_type: str | None = None
    error_message: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    # Recorded for humans and flamegraphs. NEVER part of equivalence.
    wall_start_ns: int = 0
    wall_end_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["seam"] = self.seam.value
        d["status"] = self.status.value
        d["record_type"] = "event"
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> TraceEvent:
        d = {k: v for k, v in d.items() if k != "record_type"}
        d = dict(d)
        d["seam"] = SeamKind(d["seam"])
        d["status"] = EventStatus(d["status"])
        return TraceEvent(**d)

    def equivalence_view(self) -> dict[str, Any]:
        """The projection that defines 'same execution'.

        Excludes wall clock, event_id, and parent_id (both are uuids that churn
        per run). Two runs are equivalent iff these projections match in order.
        """
        return {
            "seq": self.seq,
            "seam": self.seam.value,
            "call_site": self.call_site,
            "ordinal": self.ordinal,
            "agent_id": self.agent_id,
            "lamport": self.lamport,
            "input_digest": self.input_digest,
            "output_ref": self.output_ref,
            "status": self.status.value,
            "error_type": self.error_type,
        }

    def event_digest(self) -> str:
        return digest_obj(self.equivalence_view())


def trace_digest(events: Iterable[TraceEvent]) -> str:
    """Merkle-ish fold over the run. One number that answers 'same run?'."""
    acc = hashlib.blake2b(digest_size=32)
    acc.update(SCHEMA_VERSION.encode())
    for ev in events:
        acc.update(ev.event_digest().encode())
    return f"{DIGEST_ALGO}:{acc.hexdigest()}"

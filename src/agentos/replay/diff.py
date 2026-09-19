"""Structural diff of two hermetic traces.

Two passes, in this order:

1. Per `agent_id`, align the agent's events (ordered by lamport, then seq) with
   `difflib.SequenceMatcher`. Equality is the identity key
   `(seam, call_site, agent_id)` only. Ordinal and `input_digest` are
   deliberately out of the key: aligning on ordinal turns one insertion at a
   call site into N `input_changed` reports, and aligning on the digest turns a
   changed prompt into a delete-plus-insert instead of `input_changed`.
2. Each aligned pair is classified from `equivalence_view`. Positional fields
   (`seq`, `ordinal`, `lamport`) may shift because of insertions elsewhere;
   that is not a semantic change. Unaligned left events are `deleted`,
   unaligned right events are `inserted`.

The JSON emitted by `DiffReport.to_dict()` is a contract. `agentos bisect`
parses it; do not rename or remove fields without a migration.

Comparison uses digests only. Blob payloads are loaded only by the human
renderer, and only at the first divergence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, EventStatus, RunHeader, TraceEvent
from .store import BlobStore, TraceReader

__all__ = [
    "Change",
    "ChangeKind",
    "DiffReport",
    "Divergence",
    "IncomparableError",
    "compare_paths",
    "compare_readers",
    "diff_events",
    "render_human",
]


class ChangeKind(str, Enum):
    UNCHANGED = "unchanged"
    INPUT_CHANGED = "input_changed"
    OUTPUT_CHANGED = "output_changed"
    STATUS_CHANGED = "status_changed"
    INSERTED = "inserted"
    DELETED = "deleted"


class IncomparableError(ValueError):
    """The two traces cannot be compared. The CLI maps this to exit 125."""


# Positional artifacts of record order. They move when an earlier event is
# inserted or deleted; treating them as semantic changes is the bug this
# aligner exists to avoid.
_POSITIONAL_VIEW_KEYS = frozenset({"seq", "ordinal", "lamport"})


def identity_key(event: TraceEvent) -> tuple[str, str, str]:
    """Alignment equality. Not ordinal, not input_digest."""
    return (event.seam.value, event.call_site, event.agent_id)


def _order_key(
    left_seq: int | None,
    right_seq: int | None,
    agent_id: str = "",
    kind: str = "",
) -> tuple[int, int, str, str]:
    """Global report order: earliest cause first.

    Prefer the left (good) seq when both exist, so an insertion is placed at
    its right-seq and a substitution is placed at the good run's seq.
    """
    primary = left_seq if left_seq is not None else (
        right_seq if right_seq is not None else -1
    )
    secondary = right_seq if right_seq is not None else -1
    return (primary, secondary, agent_id, kind)


@dataclass(frozen=True, slots=True)
class Change:
    """One aligned difference. Unchanged pairs are counted, not listed."""

    kind: ChangeKind
    seq: int
    agent_id: str
    seam: str
    call_site: str
    ordinal: int | None
    left_seq: int | None
    right_seq: int | None
    left: dict[str, Any] | None
    right: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "seq": self.seq,
            "agent_id": self.agent_id,
            "seam": self.seam,
            "call_site": self.call_site,
            "ordinal": self.ordinal,
            "left_seq": self.left_seq,
            "right_seq": self.right_seq,
            "left": self.left,
            "right": self.right,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Change:
        try:
            return Change(
                kind=ChangeKind(d["kind"]),
                seq=int(d["seq"]),
                agent_id=str(d["agent_id"]),
                seam=str(d["seam"]),
                call_site=str(d["call_site"]),
                ordinal=None if d["ordinal"] is None else int(d["ordinal"]),
                left_seq=None if d["left_seq"] is None else int(d["left_seq"]),
                right_seq=None if d["right_seq"] is None else int(d["right_seq"]),
                left=None if d["left"] is None else dict(d["left"]),
                right=None if d["right"] is None else dict(d["right"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"Change JSON missing field {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class Divergence:
    """The first semantic difference, with a message that localizes it."""

    seq: int
    kind: ChangeKind
    seam: str
    call_site: str
    agent_id: str
    ordinal: int | None
    left_seq: int | None
    right_seq: int | None
    last_common_seq: int | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind.value,
            "seam": self.seam,
            "call_site": self.call_site,
            "agent_id": self.agent_id,
            "ordinal": self.ordinal,
            "left_seq": self.left_seq,
            "right_seq": self.right_seq,
            "last_common_seq": self.last_common_seq,
            "message": self.message,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Divergence:
        try:
            return Divergence(
                seq=int(d["seq"]),
                kind=ChangeKind(d["kind"]),
                seam=str(d["seam"]),
                call_site=str(d["call_site"]),
                agent_id=str(d["agent_id"]),
                ordinal=None if d["ordinal"] is None else int(d["ordinal"]),
                left_seq=None if d["left_seq"] is None else int(d["left_seq"]),
                right_seq=None if d["right_seq"] is None else int(d["right_seq"]),
                last_common_seq=(
                    None
                    if d["last_common_seq"] is None
                    else int(d["last_common_seq"])
                ),
                message=str(d["message"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"Divergence JSON missing field {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class DiffReport:
    """Result of aligning two traces. `--json` emits `to_dict()` verbatim."""

    identical: bool
    first_divergence: Divergence | None
    changes: tuple[Change, ...]
    n_unchanged: int
    n_input_changed: int
    n_output_changed: int
    n_status_changed: int
    n_inserted: int
    n_deleted: int
    left_agent_ids: tuple[str, ...]
    right_agent_ids: tuple[str, ...]
    agent_ids_differ: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "identical": self.identical,
            "first_divergence": (
                None
                if self.first_divergence is None
                else self.first_divergence.to_dict()
            ),
            "changes": [c.to_dict() for c in self.changes],
            "n_unchanged": self.n_unchanged,
            "n_input_changed": self.n_input_changed,
            "n_output_changed": self.n_output_changed,
            "n_status_changed": self.n_status_changed,
            "n_inserted": self.n_inserted,
            "n_deleted": self.n_deleted,
            "left_agent_ids": list(self.left_agent_ids),
            "right_agent_ids": list(self.right_agent_ids),
            "agent_ids_differ": self.agent_ids_differ,
            "warnings": list(self.warnings),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> DiffReport:
        try:
            raw_div = d["first_divergence"]
            return DiffReport(
                identical=bool(d["identical"]),
                first_divergence=(
                    None if raw_div is None else Divergence.from_dict(raw_div)
                ),
                changes=tuple(Change.from_dict(c) for c in d["changes"]),
                n_unchanged=int(d["n_unchanged"]),
                n_input_changed=int(d["n_input_changed"]),
                n_output_changed=int(d["n_output_changed"]),
                n_status_changed=int(d["n_status_changed"]),
                n_inserted=int(d["n_inserted"]),
                n_deleted=int(d["n_deleted"]),
                left_agent_ids=tuple(str(x) for x in d["left_agent_ids"]),
                right_agent_ids=tuple(str(x) for x in d["right_agent_ids"]),
                agent_ids_differ=bool(d["agent_ids_differ"]),
                warnings=tuple(str(x) for x in d.get("warnings", ())),
            )
        except KeyError as exc:
            raise ValueError(
                f"DiffReport JSON missing field {exc.args[0]!r}"
            ) from exc


def classify_pair(left: TraceEvent, right: TraceEvent) -> ChangeKind:
    """Classify an aligned pair from equivalence_view, ignoring positional drift.

    Priority: input, then output, then status. A changed prompt that also
    produced a different completion is `input_changed` — that is the cause.
    `status_changed` is reserved for same input and output with a different
    status or error_type (the isolated case the tests pin down).
    """
    lv = left.equivalence_view()
    rv = right.equivalence_view()
    if lv == rv:
        return ChangeKind.UNCHANGED
    if left.input_digest != right.input_digest:
        return ChangeKind.INPUT_CHANGED
    if left.output_ref != right.output_ref:
        return ChangeKind.OUTPUT_CHANGED
    if left.status != right.status or left.error_type != right.error_type:
        return ChangeKind.STATUS_CHANGED
    # Only seq / ordinal / lamport (and identical identity keys) differ.
    leftover = {
        k: lv[k]
        for k in lv
        if k not in _POSITIONAL_VIEW_KEYS and lv[k] != rv[k]
    }
    if leftover:
        # Defensive: a new equivalence_view field must not silently become
        # "unchanged". Classify as output_changed — it is the closest bucket
        # for "same call, different recorded payload".
        return ChangeKind.OUTPUT_CHANGED
    return ChangeKind.UNCHANGED


def _align_agent(
    left: Sequence[TraceEvent],
    right: Sequence[TraceEvent],
) -> list[tuple[TraceEvent | None, TraceEvent | None]]:
    left_s = sorted(left, key=lambda e: (e.lamport, e.seq))
    right_s = sorted(right, key=lambda e: (e.lamport, e.seq))
    matcher = SequenceMatcher(
        a=[identity_key(e) for e in left_s],
        b=[identity_key(e) for e in right_s],
        autojunk=False,
    )
    pairs: list[tuple[TraceEvent | None, TraceEvent | None]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                pairs.append((left_s[i1 + offset], right_s[j1 + offset]))
        elif tag == "delete":
            for event in left_s[i1:i2]:
                pairs.append((event, None))
        elif tag == "insert":
            for event in right_s[j1:j2]:
                pairs.append((None, event))
        else:
            # replace: identity keys differ, so this is delete + insert, not
            # a field-level change on a matched step.
            for event in left_s[i1:i2]:
                pairs.append((event, None))
            for event in right_s[j1:j2]:
                pairs.append((None, event))
    return pairs


def _change_from_pair(
    left: TraceEvent | None, right: TraceEvent | None
) -> tuple[ChangeKind, Change | None]:
    if left is None and right is None:
        raise ValueError("internal error: align produced an empty pair")
    if left is None:
        assert right is not None
        return ChangeKind.INSERTED, _unaligned_change(ChangeKind.INSERTED, right, side="right")
    if right is None:
        return ChangeKind.DELETED, _unaligned_change(ChangeKind.DELETED, left, side="left")
    kind = classify_pair(left, right)
    if kind is ChangeKind.UNCHANGED:
        return kind, None
    return kind, _aligned_change(kind, left, right)


def _unaligned_change(
    kind: ChangeKind, event: TraceEvent, *, side: str
) -> Change:
    view = event.equivalence_view()
    left_seq = event.seq if side == "left" else None
    right_seq = event.seq if side == "right" else None
    return Change(
        kind=kind,
        seq=event.seq,
        agent_id=event.agent_id,
        seam=event.seam.value,
        call_site=event.call_site,
        ordinal=event.ordinal,
        left_seq=left_seq,
        right_seq=right_seq,
        left=view if side == "left" else None,
        right=view if side == "right" else None,
    )


def _aligned_change(
    kind: ChangeKind, left: TraceEvent, right: TraceEvent
) -> Change:
    return Change(
        kind=kind,
        seq=left.seq,
        agent_id=left.agent_id,
        seam=left.seam.value,
        call_site=left.call_site,
        ordinal=left.ordinal,
        left_seq=left.seq,
        right_seq=right.seq,
        left=left.equivalence_view(),
        right=right.equivalence_view(),
    )


def _localize(
    change: Change,
    *,
    last_common_seq: int | None,
    left_agents: tuple[str, ...],
    right_agents: tuple[str, ...],
) -> str:
    ordinal = (
        f"ordinal {change.ordinal}"
        if change.ordinal is not None
        else "no ordinal"
    )
    site = f"{change.seam}/{change.call_site}"
    msg = (
        f"First divergence at seq={change.seq} ({change.kind.value}, "
        f"{site}, {ordinal}"
    )
    if change.agent_id != "root":
        msg += f", agent {change.agent_id}"
    msg += ")."
    if last_common_seq is None:
        msg += (
            " There is no common prefix; the first event is already "
            "a divergence."
        )
    else:
        msg += f" Last common event: seq={last_common_seq}."
    if left_agents != right_agents:
        msg += (
            f" Agent ids differ: left={list(left_agents)} "
            f"right={list(right_agents)}."
        )
    return msg


def diff_events(
    left: Sequence[TraceEvent],
    right: Sequence[TraceEvent],
) -> DiffReport:
    """Align two event lists. No header guards; no blob I/O."""
    left_by: dict[str, list[TraceEvent]] = {}
    right_by: dict[str, list[TraceEvent]] = {}
    for event in left:
        left_by.setdefault(event.agent_id, []).append(event)
    for event in right:
        right_by.setdefault(event.agent_id, []).append(event)

    left_agents = tuple(sorted(left_by))
    right_agents = tuple(sorted(right_by))
    agent_ids_differ = left_agents != right_agents

    n_unchanged = 0
    n_input_changed = 0
    n_output_changed = 0
    n_status_changed = 0
    n_inserted = 0
    n_deleted = 0
    changes: list[Change] = []
    unchanged_keys: list[tuple[int, int, str, str]] = []

    for agent_id in sorted(set(left_by) | set(right_by)):
        pairs = _align_agent(left_by.get(agent_id, []), right_by.get(agent_id, []))
        for left_ev, right_ev in pairs:
            kind, change = _change_from_pair(left_ev, right_ev)
            if kind is ChangeKind.UNCHANGED:
                n_unchanged += 1
                assert left_ev is not None and right_ev is not None
                unchanged_keys.append(
                    _order_key(left_ev.seq, right_ev.seq, agent_id, kind.value)
                )
                continue
            assert change is not None
            changes.append(change)
            if kind is ChangeKind.INPUT_CHANGED:
                n_input_changed += 1
            elif kind is ChangeKind.OUTPUT_CHANGED:
                n_output_changed += 1
            elif kind is ChangeKind.STATUS_CHANGED:
                n_status_changed += 1
            elif kind is ChangeKind.INSERTED:
                n_inserted += 1
            else:
                n_deleted += 1

    changes.sort(
        key=lambda c: _order_key(c.left_seq, c.right_seq, c.agent_id, c.kind.value)
    )

    first: Divergence | None = None
    if changes:
        head = changes[0]
        head_key = _order_key(
            head.left_seq, head.right_seq, head.agent_id, head.kind.value
        )
        prior = [k for k in unchanged_keys if k < head_key]
        last_common_seq = max(prior)[0] if prior else None
        first = Divergence(
            seq=head.seq,
            kind=head.kind,
            seam=head.seam,
            call_site=head.call_site,
            agent_id=head.agent_id,
            ordinal=head.ordinal,
            left_seq=head.left_seq,
            right_seq=head.right_seq,
            last_common_seq=last_common_seq,
            message=_localize(
                head,
                last_common_seq=last_common_seq,
                left_agents=left_agents,
                right_agents=right_agents,
            ),
        )

    return DiffReport(
        identical=not changes,
        first_divergence=first,
        changes=tuple(changes),
        n_unchanged=n_unchanged,
        n_input_changed=n_input_changed,
        n_output_changed=n_output_changed,
        n_status_changed=n_status_changed,
        n_inserted=n_inserted,
        n_deleted=n_deleted,
        left_agent_ids=left_agents,
        right_agent_ids=right_agents,
        agent_ids_differ=agent_ids_differ,
    )


def _schema_major(version: object) -> str:
    return str(version if version is not None else "0").split(".")[0]


def _is_tainted(header: RunHeader, events: Sequence[TraceEvent]) -> bool:
    if header.policy == "lenient":
        return True
    return any(event.status is EventStatus.TAINTED for event in events)


def _fmt_argv(target: list[str] | None) -> str:
    if target is None:
        return "(none)"
    return " ".join(target) if target else "(empty)"


def _schema_tuple(version: object) -> tuple[int, int, int]:
    parts = str(version if version is not None else "0").split(".")
    nums: list[int] = []
    for part in parts[:3]:
        try:
            nums.append(int(part))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def _input_storage_warning(side: str, header: RunHeader) -> str | None:
    """Loud about the gap, never incomparable. Missing blobs are not a 125."""
    if header.stores_input_blobs:
        return None
    if _schema_tuple(header.schema_version) < (0, 4, 0):
        return f"{side} trace predates input storage; comparing digests only"
    return f"{side} trace did not store input blobs; comparing digests only"


def _comparison_warnings(left: RunHeader, right: RunHeader) -> tuple[str, ...]:
    warnings: list[str] = []
    if left.git_sha != right.git_sha:
        warnings.append(
            f"git sha differs (left={left.git_sha}, right={right.git_sha}). "
            "This is expected during bisect; proceeding."
        )
    if left.schema_version != right.schema_version:
        warnings.append(
            f"schema minor versions differ (left={left.schema_version}, "
            f"right={right.schema_version}). Proceeding; fields added since "
            "the older schema take their defaults."
        )
    if left.target != right.target:
        warnings.append(
            "target argv differs:\n"
            f"  left:  {_fmt_argv(left.target)}\n"
            f"  right: {_fmt_argv(right.target)}\n"
            "This is usually a mistake (comparing runs of different programs)."
        )
    for side, header in (("left", left), ("right", right)):
        note = _input_storage_warning(side, header)
        if note is not None:
            warnings.append(note)
    return tuple(warnings)


def _guard_readers(left: TraceReader, right: TraceReader) -> None:
    left_major = _schema_major(left.header.schema_version)
    right_major = _schema_major(right.header.schema_version)
    if left_major != right_major:
        raise IncomparableError(
            f"schema major versions differ (left={left.header.schema_version}, "
            f"right={right.header.schema_version}). Readers refuse to guess "
            "across majors; run `agentos trace migrate`."
        )

    if dict(left.header.seam_codecs) != dict(right.header.seam_codecs):
        raise IncomparableError(
            "seam codec fingerprints differ: the two runs digested different "
            f"field sets (left={dict(left.header.seam_codecs)}, "
            f"right={dict(right.header.seam_codecs)}). Comparing input digests "
            "would be meaningless. Re-record both traces with the same build."
        )

    tainted_sides = [
        side
        for side, reader in (("left", left), ("right", right))
        if _is_tainted(reader.header, reader.events)
    ]
    if tainted_sides:
        which = " and ".join(tainted_sides)
        raise IncomparableError(
            f"{which} trace is tainted (recorded under LENIENT or contains a "
            "tainted event). A tainted run is not valid input for any "
            "comparison."
        )


def compare_readers(left: TraceReader, right: TraceReader) -> DiffReport:
    """Header guards, then `diff_events`. Still no blob I/O."""
    _guard_readers(left, right)
    report = diff_events(left.events, right.events)
    warnings = _comparison_warnings(left.header, right.header)
    if not warnings:
        return report
    return DiffReport(
        identical=report.identical,
        first_divergence=report.first_divergence,
        changes=report.changes,
        n_unchanged=report.n_unchanged,
        n_input_changed=report.n_input_changed,
        n_output_changed=report.n_output_changed,
        n_status_changed=report.n_status_changed,
        n_inserted=report.n_inserted,
        n_deleted=report.n_deleted,
        left_agent_ids=report.left_agent_ids,
        right_agent_ids=report.right_agent_ids,
        agent_ids_differ=report.agent_ids_differ,
        warnings=warnings,
    )


def _require_file(side: str, path: Path) -> None:
    if not path.is_file():
        raise IncomparableError(f"{side} trace is missing: {path}")


def _read_header_raw(side: str, path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IncomparableError(f"{side} trace is unreadable: {exc}") from exc
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise IncomparableError(
                f"{side} trace is unreadable: invalid JSON ({exc})"
            ) from exc
        if not isinstance(rec, dict):
            raise IncomparableError(
                f"{side} trace is unreadable: first record is not an object"
            )
        if rec.get("record_type") == "header":
            return rec
        raise IncomparableError(
            f"{side} trace is unreadable: first record is not a header"
        )
    raise IncomparableError(f"{side} trace is unreadable: no header record")


def _load_reader(side: str, path: Path) -> TraceReader:
    try:
        return TraceReader(path)
    except (ValueError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise IncomparableError(
            f"{side} trace is unreadable ({type(exc).__name__}: {exc})"
        ) from exc


def compare_paths(left_path: str | Path, right_path: str | Path) -> DiffReport:
    """Load two jsonl traces, refuse incomparable pairs, then align."""
    left_path = Path(left_path)
    right_path = Path(right_path)
    _require_file("left", left_path)
    _require_file("right", right_path)
    left_raw = _read_header_raw("left", left_path)
    right_raw = _read_header_raw("right", right_path)
    if _schema_major(left_raw.get("schema_version")) != _schema_major(
        right_raw.get("schema_version")
    ):
        raise IncomparableError(
            "schema major versions differ "
            f"(left={left_raw.get('schema_version')}, "
            f"right={right_raw.get('schema_version')}). "
            "Readers refuse to guess across majors; run `agentos trace migrate`."
        )
    # Still refuse a pair whose shared major this build cannot read.
    current_major = _schema_major(SCHEMA_VERSION)
    for side, raw in (("left", left_raw), ("right", right_raw)):
        major = _schema_major(raw.get("schema_version"))
        if major != current_major:
            raise IncomparableError(
                f"{side} trace schema major version {major} is not readable by "
                f"{SCHEMA_VERSION}; run `agentos trace migrate`"
            )
    return compare_readers(_load_reader("left", left_path), _load_reader("right", right_path))


def _pretty_json_lines(obj: Any) -> list[str]:
    text = json.dumps(
        obj,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return [line + "\n" for line in text.splitlines()]


def _load_input(
    store: BlobStore | None,
    digest: str | None,
    *,
    stored: bool,
) -> Any | None:
    if not stored or store is None or not digest:
        return None
    try:
        return store.get_obj(digest)
    except KeyError:
        return None


def _context_window(
    events: Sequence[TraceEvent], locus_seq: int | None, n: int
) -> list[TraceEvent]:
    ordered = sorted(events, key=lambda e: e.seq)
    if not ordered or n < 0:
        return []
    if locus_seq is None:
        idx = 0
    else:
        idx = 0
        found = False
        for i, event in enumerate(ordered):
            if event.seq == locus_seq:
                idx = i
                found = True
                break
            if event.seq > locus_seq:
                idx = i
                found = True
                break
        if not found:
            idx = len(ordered) - 1
    start = max(0, idx - n)
    end = min(len(ordered), idx + n + 1)
    return ordered[start:end]


def _format_event_line(event: TraceEvent, *, mark: bool) -> str:
    prefix = ">" if mark else " "
    return (
        f"{prefix} {event.seq:>4}  {event.seam.value:<10}{event.call_site:<24}"
        f"{event.ordinal:>4}  {event.status.value:<8}{event.agent_id}"
    )


def _input_diff_block(
    report: DiffReport,
    left_blobs: BlobStore | None,
    right_blobs: BlobStore | None,
    *,
    left_stores_inputs: bool,
    right_stores_inputs: bool,
) -> str:
    div = report.first_divergence
    if div is None or not report.changes:
        return ""
    head = report.changes[0]
    left_digest = None if head.left is None else head.left.get("input_digest")
    right_digest = None if head.right is None else head.right.get("input_digest")
    left_obj = _load_input(
        left_blobs,
        left_digest if isinstance(left_digest, str) else None,
        stored=left_stores_inputs,
    )
    right_obj = _load_input(
        right_blobs,
        right_digest if isinstance(right_digest, str) else None,
        stored=right_stores_inputs,
    )
    lines = ["", "input at the divergence:"]
    if left_obj is None and right_obj is None:
        lines.append(
            "  input blobs are not in the store; comparison used digests only "
            f"(left={left_digest}, right={right_digest})."
        )
        return "\n".join(lines)
    fromfile = f"good seq={div.left_seq}"
    tofile = f"bad seq={div.right_seq}"
    left_lines = _pretty_json_lines(left_obj) if left_obj is not None else []
    right_lines = _pretty_json_lines(right_obj) if right_obj is not None else []
    rendered = "".join(
        unified_diff(left_lines, right_lines, fromfile=fromfile, tofile=tofile)
    )
    if not rendered:
        lines.append("  (canonical inputs are identical at this step)")
        return "\n".join(lines)
    return "\n".join(lines) + "\n" + rendered.rstrip("\n")


def render_human(
    report: DiffReport,
    left_events: Sequence[TraceEvent],
    right_events: Sequence[TraceEvent],
    *,
    context: int = 3,
    left_blobs: BlobStore | None = None,
    right_blobs: BlobStore | None = None,
    left_stores_inputs: bool = False,
    right_stores_inputs: bool = False,
) -> str:
    """First divergence, N events of context each side, unified input diff."""
    if report.identical:
        return f"identical: {report.n_unchanged} event(s), 0 changes"

    div = report.first_divergence
    assert div is not None
    parts = [div.message]
    if context < 0:
        context = 0

    parts.append("")
    parts.append(f"good (context={context}):")
    left_window = _context_window(left_events, div.left_seq, context)
    if not left_window:
        parts.append("  (no events)")
    else:
        for event in left_window:
            parts.append(
                _format_event_line(event, mark=event.seq == div.left_seq)
            )

    parts.append("")
    parts.append(f"bad (context={context}):")
    right_window = _context_window(right_events, div.right_seq, context)
    if not right_window:
        parts.append("  (no events)")
    else:
        for event in right_window:
            parts.append(
                _format_event_line(event, mark=event.seq == div.right_seq)
            )

    parts.append(
        _input_diff_block(
            report,
            left_blobs,
            right_blobs,
            left_stores_inputs=left_stores_inputs,
            right_stores_inputs=right_stores_inputs,
        )
    )
    return "\n".join(parts).rstrip() + "\n"

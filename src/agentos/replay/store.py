"""Trace persistence: content-addressed blobs plus an append-only JSONL log.

Layout on disk:

    <root>/
      blobs/<aa>/<bb>/<digest>.bin[.gz]     # deduped payloads, immutable
      runs/<run_id>.jsonl                   # header line, then one line per event

Content addressing is not a storage micro-optimization. It is what makes trace
diff cheap (compare 32 bytes, not two megabytes of prompt) and what keeps a
hundred-run bisect corpus from exploding, since every run in the corpus shares
the same system prompt blob.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

from .schema import (
    RunHeader,
    TraceEvent,
    canonical_json,
    digest_bytes,
)

GZIP_THRESHOLD = 4096  # bytes; below this, gzip costs more than it saves


class BlobStore:
    """Immutable, deduplicating, content-addressed byte store."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root) / "blobs"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str, compressed: bool) -> Path:
        hexpart = digest.split(":", 1)[1]
        suffix = ".bin.gz" if compressed else ".bin"
        return self.root / hexpart[:2] / hexpart[2:4] / f"{hexpart}{suffix}"

    def put(self, data: bytes) -> str:
        digest = digest_bytes(data)
        compressed = len(data) >= GZIP_THRESHOLD
        path = self._path(digest, compressed)
        if path.exists():
            return digest  # already stored; writes are idempotent by construction
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = gzip.compress(data, mtime=0) if compressed else data
        tmp.write_bytes(payload)
        tmp.replace(path)  # atomic, so a crashed record leaves no torn blob
        return digest

    def put_obj(self, obj: Any) -> str:
        return self.put(canonical_json(obj))

    def get(self, digest: str) -> bytes:
        plain, gz = self._path(digest, False), self._path(digest, True)
        if plain.exists():
            return plain.read_bytes()
        if gz.exists():
            return gzip.decompress(gz.read_bytes())
        raise KeyError(f"blob not found: {digest}")

    def get_obj(self, digest: str) -> Any:
        return json.loads(self.get(digest))

    def has(self, digest: str) -> bool:
        return self._path(digest, False).exists() or self._path(digest, True).exists()


class TraceWriter:
    """Append-only JSONL writer. Crash-safe by line, flushed per event."""

    def __init__(self, root: str | os.PathLike[str], header: RunHeader) -> None:
        self.root = Path(root)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.path = self.root / "runs" / f"{header.run_id}.jsonl"
        self.blobs = BlobStore(root)
        self._fh = self.path.open("w", encoding="utf-8")
        self._write(header.to_dict())
        self._seq = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _write(self, record: dict[str, Any]) -> None:
        self._fh.write(canonical_json(record).decode("utf-8") + "\n")
        self._fh.flush()

    def append(self, event: TraceEvent) -> None:
        self._write(event.to_dict())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TraceReader:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.root = self.path.parent.parent
        self.blobs = BlobStore(self.root)
        self.header, self._events = self._load()

    def _load(self) -> tuple[RunHeader, list[TraceEvent]]:
        header: RunHeader | None = None
        events: list[TraceEvent] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("record_type") == "header":
                    header = RunHeader.from_dict(rec)
                else:
                    events.append(TraceEvent.from_dict(rec))
        if header is None:
            raise ValueError(f"trace {self.path} has no header record")
        return header, events

    @property
    def events(self) -> list[TraceEvent]:
        return self._events

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self._events)

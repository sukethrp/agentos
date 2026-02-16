"""Feedback Store — collect, persist, and query user feedback.

Every feedback entry ties a user reaction (thumbs up/down, star rating,
text correction) back to the *original query + response* so the analyzer
and prompt optimizer can learn from it.

Storage is JSON-file-backed — no database required.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ── Models ───────────────────────────────────────────────────────────────────

class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"            # 1-5 stars
    CORRECTION = "correction"    # user provides the correct answer
    COMMENT = "comment"          # free-text feedback


class FeedbackEntry(BaseModel):
    """A single piece of user feedback."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_name: str = ""
    feedback_type: FeedbackType
    query: str                          # the original user query
    response: str                       # the agent's response
    rating: float = 0.0                 # 1-5 stars (0 = not rated)
    correction: str = ""                # what the response *should* have been
    comment: str = ""                   # free-text note
    topic: str = ""                     # auto-detected or user-tagged topic
    tools_used: list[str] = Field(default_factory=list)
    model: str = ""
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Derived
    @property
    def is_positive(self) -> bool:
        if self.feedback_type == FeedbackType.THUMBS_UP:
            return True
        if self.feedback_type == FeedbackType.THUMBS_DOWN:
            return False
        if self.feedback_type == FeedbackType.RATING:
            return self.rating >= 4
        return self.correction == ""

    @property
    def quality_score(self) -> float:
        """Normalise to 0-10 scale for compatibility with evaluator scores."""
        if self.feedback_type == FeedbackType.THUMBS_UP:
            return 9.0
        if self.feedback_type == FeedbackType.THUMBS_DOWN:
            return 2.0
        if self.feedback_type == FeedbackType.RATING:
            return self.rating * 2        # 1-5 → 2-10
        if self.feedback_type == FeedbackType.CORRECTION:
            return 3.0                    # needed correction → poor
        return 5.0                        # neutral comment


# ── Store ────────────────────────────────────────────────────────────────────

class FeedbackStore:
    """JSON-file-backed feedback store with query helpers."""

    def __init__(self, data_dir: str = "./agent_data/learning") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.data_dir / "feedback.json"
        self._entries: list[FeedbackEntry] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._entries = [FeedbackEntry(**e) for e in raw]
            except Exception:
                self._entries = []

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([e.model_dump() for e in self._entries], indent=2, default=str)
        )

    # ── Write ────────────────────────────────────────────────────────────

    def add(self, entry: FeedbackEntry) -> FeedbackEntry:
        self._entries.append(entry)
        self._save()
        return entry

    def thumbs_up(
        self, query: str, response: str, agent_name: str = "", **kw: Any,
    ) -> FeedbackEntry:
        return self.add(FeedbackEntry(
            feedback_type=FeedbackType.THUMBS_UP,
            query=query, response=response, agent_name=agent_name, **kw,
        ))

    def thumbs_down(
        self, query: str, response: str, agent_name: str = "",
        comment: str = "", **kw: Any,
    ) -> FeedbackEntry:
        return self.add(FeedbackEntry(
            feedback_type=FeedbackType.THUMBS_DOWN,
            query=query, response=response, agent_name=agent_name,
            comment=comment, **kw,
        ))

    def rate(
        self, query: str, response: str, rating: float,
        agent_name: str = "", **kw: Any,
    ) -> FeedbackEntry:
        return self.add(FeedbackEntry(
            feedback_type=FeedbackType.RATING,
            query=query, response=response, rating=max(1, min(rating, 5)),
            agent_name=agent_name, **kw,
        ))

    def correct(
        self, query: str, response: str, correction: str,
        agent_name: str = "", **kw: Any,
    ) -> FeedbackEntry:
        return self.add(FeedbackEntry(
            feedback_type=FeedbackType.CORRECTION,
            query=query, response=response, correction=correction,
            agent_name=agent_name, **kw,
        ))

    def comment(
        self, query: str, response: str, comment: str,
        agent_name: str = "", **kw: Any,
    ) -> FeedbackEntry:
        return self.add(FeedbackEntry(
            feedback_type=FeedbackType.COMMENT,
            query=query, response=response, comment=comment,
            agent_name=agent_name, **kw,
        ))

    # ── Read / query ─────────────────────────────────────────────────────

    def all(self) -> list[FeedbackEntry]:
        return list(self._entries)

    def positive(self) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.is_positive]

    def negative(self) -> list[FeedbackEntry]:
        return [e for e in self._entries if not e.is_positive]

    def by_agent(self, agent_name: str) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.agent_name == agent_name]

    def by_topic(self, topic: str) -> list[FeedbackEntry]:
        t = topic.lower()
        return [e for e in self._entries if t in e.topic.lower()]

    def by_type(self, ft: FeedbackType) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.feedback_type == ft]

    def corrections(self) -> list[FeedbackEntry]:
        return self.by_type(FeedbackType.CORRECTION)

    def recent(self, n: int = 20) -> list[FeedbackEntry]:
        return sorted(self._entries, key=lambda e: -e.timestamp)[:n]

    def since(self, ts: float) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.timestamp >= ts]

    def search(self, query: str) -> list[FeedbackEntry]:
        q = query.lower()
        return [
            e for e in self._entries
            if q in e.query.lower() or q in e.response.lower()
            or q in e.comment.lower() or q in e.topic.lower()
        ]

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        total = len(self._entries)
        pos = sum(1 for e in self._entries if e.is_positive)
        neg = total - pos
        avg_rating = 0.0
        rated = [e for e in self._entries if e.feedback_type == FeedbackType.RATING]
        if rated:
            avg_rating = sum(e.rating for e in rated) / len(rated)

        by_type: dict[str, int] = defaultdict(int)
        for e in self._entries:
            by_type[e.feedback_type.value] += 1

        topics: dict[str, int] = defaultdict(int)
        for e in self._entries:
            if e.topic:
                topics[e.topic] += 1

        return {
            "total": total,
            "positive": pos,
            "negative": neg,
            "positive_rate": round(pos / max(total, 1) * 100, 1),
            "avg_rating": round(avg_rating, 2),
            "corrections": sum(1 for e in self._entries if e.feedback_type == FeedbackType.CORRECTION),
            "by_type": dict(by_type),
            "top_topics": dict(sorted(topics.items(), key=lambda x: -x[1])[:10]),
        }

    def clear(self) -> None:
        self._entries = []
        self._save()


# ── Default singleton ────────────────────────────────────────────────────────

_default_store: FeedbackStore | None = None


def get_feedback_store() -> FeedbackStore:
    global _default_store
    if _default_store is None:
        _default_store = FeedbackStore()
    return _default_store

"""Per-user usage tracking for AgentOS.

Tracks:
  - number of queries
  - tokens used
  - total cost (USD)
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Literal


Period = Literal["day", "week", "month"]


@dataclass
class UsageRecord:
    user_id: str
    timestamp: float
    tokens: int
    cost: float

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "tokens": self.tokens,
            "cost": self.cost,
        }


@dataclass
class UsageSummary:
    user_id: str
    queries: int
    tokens: int
    cost: float

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "queries": self.queries,
            "tokens": self.tokens,
            "cost": self.cost,
        }


class UsageTracker:
    """JSON file-backed usage tracker."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write({})

    # ── IO helpers ──

    def _read(self) -> Dict[str, List[dict]]:
        with self._lock:
            if not os.path.exists(self.path):
                return {}
            with open(self.path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    return {}
                except json.JSONDecodeError:
                    return {}

    def _write(self, data: Dict[str, List[dict]]) -> None:
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    # ── Public API ──

    def log_usage(self, user_id: str, tokens: int, cost: float) -> None:
        """Record a single usage entry for a user."""
        if not user_id:
            return
        record = UsageRecord(
            user_id=user_id,
            timestamp=time.time(),
            tokens=int(tokens),
            cost=float(cost),
        )
        data = self._read()
        records = data.get(user_id, [])
        records.append(record.to_dict())
        # cap history per user to avoid unbounded growth
        if len(records) > 1000:
            records = records[-1000:]
        data[user_id] = records
        self._write(data)

    def _get_records(self, user_id: str) -> List[UsageRecord]:
        data = self._read()
        raw = data.get(user_id, [])
        return [UsageRecord(**r) for r in raw]

    def get_usage(self, user_id: str) -> UsageSummary:
        """Return lifetime usage summary for a user."""
        records = self._get_records(user_id)
        total_tokens = sum(r.tokens for r in records)
        total_cost = sum(r.cost for r in records)
        return UsageSummary(
            user_id=user_id,
            queries=len(records),
            tokens=total_tokens,
            cost=total_cost,
        )

    def get_usage_by_period(self, user_id: str, period: Period = "day") -> UsageSummary:
        """Return usage summary for the last day/week/month."""
        records = self._get_records(user_id)
        if not records:
            return UsageSummary(user_id=user_id, queries=0, tokens=0, cost=0.0)

        now = time.time()
        if period == "day":
            window = 60 * 60 * 24
        elif period == "week":
            window = 60 * 60 * 24 * 7
        else:
            window = 60 * 60 * 24 * 30

        filtered = [r for r in records if now - r.timestamp <= window]
        total_tokens = sum(r.tokens for r in filtered)
        total_cost = sum(r.cost for r in filtered)
        return UsageSummary(
            user_id=user_id,
            queries=len(filtered),
            tokens=total_tokens,
            cost=total_cost,
        )


def default_usage_path() -> str:
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "usage.json")


# Module-level default tracker
usage_tracker = UsageTracker(path=default_usage_path())

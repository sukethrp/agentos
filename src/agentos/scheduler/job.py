"""Scheduler Job — represents a single scheduled agent task."""

from __future__ import annotations
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class JobExecution:
    """Record of a single execution of a scheduled job."""

    started_at: float
    finished_at: float = 0.0
    status: str = "completed"
    result: str = ""
    error: str = ""
    cost_usd: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        if self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "result": self.result[:500],
            "error": self.error,
            "cost_usd": self.cost_usd,
            "tokens_used": self.tokens_used,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class Job:
    """A scheduled agent job."""

    agent_name: str
    query: str
    interval_seconds: float = 0.0
    cron_expression: str = ""
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.PENDING
    max_executions: int = 0  # 0 = unlimited
    created_at: float = field(default_factory=time.time)

    # Runtime state
    execution_count: int = 0
    last_run: float = 0.0
    next_run: float = 0.0
    history: list[JobExecution] = field(default_factory=list)

    def __post_init__(self):
        if self.next_run == 0.0:
            if self.interval_seconds > 0:
                self.next_run = time.time() + self.interval_seconds
            elif self.cron_expression:
                self.next_run = next_cron_time(self.cron_expression)

    def record_execution(
        self, result: str, cost: float = 0.0, tokens: int = 0, error: str = ""
    ) -> None:
        """Record a completed execution."""
        now = time.time()
        execution = JobExecution(
            started_at=self.last_run or now,
            finished_at=now,
            status="failed" if error else "completed",
            result=result,
            error=error,
            cost_usd=cost,
            tokens_used=tokens,
        )
        self.history.append(execution)
        self.execution_count += 1

        # Keep only last 50 executions
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def update_next_run(self) -> None:
        """Calculate the next run time."""
        if self.interval_seconds > 0:
            self.next_run = time.time() + self.interval_seconds
        elif self.cron_expression:
            self.next_run = next_cron_time(self.cron_expression)

    def is_due(self) -> bool:
        """Check if this job should run now."""
        if self.status in (JobStatus.PAUSED, JobStatus.CANCELLED):
            return False
        if self.max_executions > 0 and self.execution_count >= self.max_executions:
            return False
        return time.time() >= self.next_run

    def is_finished(self) -> bool:
        """Check if this job has completed all scheduled executions."""
        if self.status == JobStatus.CANCELLED:
            return True
        if self.max_executions > 0 and self.execution_count >= self.max_executions:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "agent_name": self.agent_name,
            "query": self.query,
            "interval_seconds": self.interval_seconds,
            "cron_expression": self.cron_expression,
            "status": self.status.value,
            "execution_count": self.execution_count,
            "max_executions": self.max_executions,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "created_at": self.created_at,
            "history": [h.to_dict() for h in self.history[-10:]],
        }


# ── Interval parsing ──

INTERVAL_PATTERN = re.compile(
    r"^(\d+)\s*(s|sec|m|min|h|hr|hour|d|day)s?$", re.IGNORECASE
)

INTERVAL_MULTIPLIERS = {
    "s": 1,
    "sec": 1,
    "m": 60,
    "min": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "d": 86400,
    "day": 86400,
}


def parse_interval(interval: str) -> float:
    """Parse an interval string like '5m', '1h', '30s', '1d' into seconds.

    Returns 0.0 if parsing fails.
    """
    match = INTERVAL_PATTERN.match(interval.strip())
    if not match:
        return 0.0
    value = int(match.group(1))
    unit = match.group(2).lower()
    multiplier = INTERVAL_MULTIPLIERS.get(unit, 0)
    return float(value * multiplier)


# ── Cron parsing (basic: minute hour day_of_month month day_of_week) ──


def next_cron_time(cron_expr: str) -> float:
    """Calculate the next run time for a basic cron expression.

    Supports: minute hour day_of_month month day_of_week
    Supports: numbers, *, and */N step syntax.

    Examples:
        "0 9 * * *"     -> 9:00 AM daily
        "*/5 * * * *"   -> every 5 minutes
        "30 */2 * * *"  -> at minute 30, every 2 hours
        "0 0 * * 1"     -> midnight every Monday
    """
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        return time.time() + 60  # fallback: 1 minute

    now = datetime.now()
    # Search forward up to 48 hours to find next match
    candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(48 * 60):  # max 48 hours of minutes
        if _cron_matches(candidate, parts):
            return candidate.timestamp()
        candidate += timedelta(minutes=1)

    # Fallback: 1 hour from now
    return time.time() + 3600


def _cron_matches(dt: datetime, parts: list[str]) -> bool:
    """Check if a datetime matches a cron expression."""
    fields = [dt.minute, dt.hour, dt.day, dt.month, dt.isoweekday() % 7]
    ranges = [
        (0, 59),  # minute
        (0, 23),  # hour
        (1, 31),  # day of month
        (1, 12),  # month
        (0, 6),  # day of week (0=Sunday)
    ]

    for i, (pattern, value, (low, high)) in enumerate(zip(parts, fields, ranges)):
        if not _field_matches(pattern, value, low, high):
            return False
    return True


def _field_matches(pattern: str, value: int, low: int, high: int) -> bool:
    """Check if a single cron field matches a value."""
    if pattern == "*":
        return True

    # Step: */N
    if pattern.startswith("*/"):
        try:
            step = int(pattern[2:])
            return step > 0 and (value - low) % step == 0
        except ValueError:
            return False

    # List: 1,3,5
    if "," in pattern:
        try:
            return value in {int(v) for v in pattern.split(",")}
        except ValueError:
            return False

    # Range: 1-5
    if "-" in pattern:
        try:
            start, end = pattern.split("-", 1)
            return int(start) <= value <= int(end)
        except ValueError:
            return False

    # Exact match
    try:
        return value == int(pattern)
    except ValueError:
        return False

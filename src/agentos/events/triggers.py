"""Event triggers — fire events in response to external stimuli.

Triggers:
    WebhookTrigger   — fires when an HTTP POST is received
    TimerTrigger     — fires at regular intervals
    AgentCompleteTrigger — fires when another agent finishes
    FileTrigger      — watches a directory for new/changed files
"""

from __future__ import annotations
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from agentos.events.bus import EventBus, event_bus


# ────────────────────────────────────────────
#  Base Trigger
# ────────────────────────────────────────────
@dataclass
class BaseTrigger:
    """Common base for all triggers."""

    name: str = ""
    bus: EventBus = field(default_factory=lambda: event_bus)
    _running: bool = field(default=False, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "running": self._running,
        }


# ────────────────────────────────────────────
#  WebhookTrigger
# ────────────────────────────────────────────
@dataclass
class WebhookTrigger(BaseTrigger):
    """Fires an event when an HTTP webhook is received.

    This trigger is "passive" — it relies on the FastAPI /api/webhook/{event_name}
    endpoint calling ``fire()`` rather than running its own HTTP server.
    """

    event_name: str = "webhook.received"

    def start(self) -> None:
        self._running = True

    def fire(self, data: dict | None = None, source: str = "webhook") -> None:
        """Called by the webhook API endpoint."""
        if not self._running:
            return
        self.bus.emit(self.event_name, data=data or {}, source=source)

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "event_name": self.event_name,
        }


# ────────────────────────────────────────────
#  TimerTrigger
# ────────────────────────────────────────────
@dataclass
class TimerTrigger(BaseTrigger):
    """Fires an event at fixed intervals.

    Args:
        interval_seconds: Seconds between firings.
        event_name: Event to emit (default "timer.fired").
        max_fires: Stop after this many firings (0 = unlimited).
    """

    interval_seconds: float = 60.0
    event_name: str = "timer.fired"
    max_fires: int = 0
    _fire_count: int = field(default=0, repr=False)

    def start(self) -> None:
        self._running = True
        self._fire_count = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            time.sleep(self.interval_seconds)
            if not self._running:
                break
            self._fire_count += 1
            self.bus.emit(
                self.event_name,
                data={
                    "fire_count": self._fire_count,
                    "trigger": self.name,
                    "interval_seconds": self.interval_seconds,
                },
                source=f"timer:{self.name}",
            )
            if 0 < self.max_fires <= self._fire_count:
                self._running = False
                break

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "event_name": self.event_name,
            "interval_seconds": self.interval_seconds,
            "fire_count": self._fire_count,
            "max_fires": self.max_fires,
        }


# ────────────────────────────────────────────
#  AgentCompleteTrigger
# ────────────────────────────────────────────
@dataclass
class AgentCompleteTrigger(BaseTrigger):
    """Fires an event when a monitored agent completes its run.

    The consuming code (or Agent.run) should call ``fire()`` after the
    agent finishes.  Alternatively, you can wrap ``agent.run`` with
    ``wrap_agent()``.
    """

    watched_agent_name: str = ""
    event_name: str = "agent.completed"

    def start(self) -> None:
        self._running = True

    def fire(self, agent_name: str, result: str, cost: float = 0.0, tokens: int = 0) -> None:
        if not self._running:
            return
        self.bus.emit(
            self.event_name,
            data={
                "agent_name": agent_name,
                "result": result[:2000],
                "cost_usd": cost,
                "tokens_used": tokens,
            },
            source=f"agent:{agent_name}",
        )

    def wrap_agent(self, agent: Any) -> Any:
        """Return a thin wrapper that emits 'agent.completed' after run()."""
        trigger = self

        class _WrappedAgent:
            def __init__(self, inner: Any):
                self._inner = inner
                self.config = inner.config
                self.events = inner.events

            def run(self, user_input: str, **kwargs):
                result = self._inner.run(user_input, **kwargs)
                # Fire the trigger
                content = result.content if hasattr(result, "content") else str(result)
                cost = sum(e.cost_usd for e in self._inner.events) if self._inner.events else 0
                tokens = sum(e.tokens_used for e in self._inner.events) if self._inner.events else 0
                trigger.fire(
                    agent_name=self._inner.config.name,
                    result=content,
                    cost=cost,
                    tokens=tokens,
                )
                return result

            def __getattr__(self, name):
                return getattr(self._inner, name)

        return _WrappedAgent(agent)

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "event_name": self.event_name,
            "watched_agent_name": self.watched_agent_name,
        }


# ────────────────────────────────────────────
#  FileTrigger
# ────────────────────────────────────────────
@dataclass
class FileTrigger(BaseTrigger):
    """Watches a directory for new or changed files and emits events.

    Args:
        watch_dir: Directory to watch.
        event_name: Event name to emit (default "file.changed").
        poll_interval: Seconds between polls.
        patterns: Glob-like extensions to include, e.g. [".txt", ".md"].
                  Empty list means all files.
    """

    watch_dir: str = "."
    event_name: str = "file.changed"
    poll_interval: float = 2.0
    patterns: list[str] = field(default_factory=list)
    _known_files: dict[str, float] = field(default_factory=dict, repr=False)

    def start(self) -> None:
        self._running = True
        # Seed known files
        self._known_files = self._scan()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _scan(self) -> dict[str, float]:
        """Return {filepath: mtime} for all matching files."""
        result: dict[str, float] = {}
        if not os.path.isdir(self.watch_dir):
            return result
        for entry in os.scandir(self.watch_dir):
            if entry.is_file():
                if self.patterns and not any(entry.name.endswith(p) for p in self.patterns):
                    continue
                result[entry.path] = entry.stat().st_mtime
        return result

    def _loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)
            if not self._running:
                break
            current = self._scan()

            # New files
            for path, mtime in current.items():
                if path not in self._known_files:
                    self.bus.emit(
                        self.event_name,
                        data={
                            "action": "created",
                            "path": path,
                            "filename": os.path.basename(path),
                        },
                        source=f"file:{self.name}",
                    )
                elif mtime > self._known_files[path]:
                    self.bus.emit(
                        self.event_name,
                        data={
                            "action": "modified",
                            "path": path,
                            "filename": os.path.basename(path),
                        },
                        source=f"file:{self.name}",
                    )

            # Deleted files
            for path in set(self._known_files) - set(current):
                self.bus.emit(
                    self.event_name,
                    data={
                        "action": "deleted",
                        "path": path,
                        "filename": os.path.basename(path),
                    },
                    source=f"file:{self.name}",
                )

            self._known_files = current

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "event_name": self.event_name,
            "watch_dir": self.watch_dir,
            "poll_interval": self.poll_interval,
            "patterns": self.patterns,
            "known_files": len(self._known_files),
        }

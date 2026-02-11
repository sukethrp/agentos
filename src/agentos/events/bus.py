"""EventBus — publish/subscribe event system for agent orchestration.

Usage:
    from agentos.events import event_bus

    event_bus.on("webhook.received", agent, "Analyze this data: {data}")
    event_bus.emit("webhook.received", {"data": "payload..."})
"""

from __future__ import annotations
import fnmatch
import io
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    """A single event that flows through the bus."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class Listener:
    """An agent registered to react to events."""

    event_pattern: str  # exact name or glob like "custom.*"
    agent: Any  # Agent instance
    query_template: str  # e.g. "Analyze: {data}"
    listener_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    execution_count: int = 0
    last_triggered: float = 0.0

    def matches(self, event_name: str) -> bool:
        """Check if this listener should react to the given event."""
        if self.event_pattern == event_name:
            return True
        return fnmatch.fnmatch(event_name, self.event_pattern)

    def build_query(self, event: Event) -> str:
        """Fill in the query template with event data."""
        query = self.query_template
        # Replace {key} with event.data[key]
        for key, value in event.data.items():
            query = query.replace(f"{{{key}}}", str(value))
        # Also support {event_name}, {event_id}, {source}
        query = query.replace("{event_name}", event.name)
        query = query.replace("{event_id}", event.event_id)
        query = query.replace("{source}", event.source)
        return query

    def to_dict(self) -> dict:
        return {
            "listener_id": self.listener_id,
            "event_pattern": self.event_pattern,
            "agent_name": getattr(self.agent, "config", None) and self.agent.config.name or "unknown",
            "query_template": self.query_template,
            "execution_count": self.execution_count,
            "last_triggered": self.last_triggered,
            "created_at": self.created_at,
        }


@dataclass
class EventLog:
    """Record of an event emission and its results."""

    event: Event
    listeners_triggered: int = 0
    results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
            "listeners_triggered": self.listeners_triggered,
            "results": self.results[-10:],
        }


class EventBus:
    """Central event bus for agent-to-agent communication.

    Supports exact event names and glob patterns:
        "webhook.received"   — exact match
        "agent.completed"    — exact match
        "custom.*"           — matches custom.anything
        "*"                  — matches everything
    """

    def __init__(self, max_concurrent: int = 5):
        self._listeners: list[Listener] = []
        self._history: list[EventLog] = []
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent
        self._active_count = 0
        self._callbacks: list[Callable] = []  # raw callbacks (non-agent)

    # ── Register / Unregister ──

    def on(self, event_pattern: str, agent: Any, query_template: str) -> str:
        """Register an agent to react when a matching event is emitted.

        Args:
            event_pattern: Event name or glob (e.g. "webhook.*", "agent.completed").
            agent: An Agent instance that will run the query.
            query_template: Template with {placeholders} filled from event data.

        Returns:
            The listener ID.
        """
        listener = Listener(
            event_pattern=event_pattern,
            agent=agent,
            query_template=query_template,
        )
        with self._lock:
            self._listeners.append(listener)
        return listener.listener_id

    def off(self, event_pattern: str, agent: Any) -> bool:
        """Unregister an agent from an event pattern.

        Returns True if a listener was removed.
        """
        with self._lock:
            before = len(self._listeners)
            agent_name = getattr(agent, "config", None) and agent.config.name or id(agent)
            self._listeners = [
                l for l in self._listeners
                if not (
                    l.event_pattern == event_pattern
                    and (getattr(l.agent, "config", None) and l.agent.config.name or id(l.agent)) == agent_name
                )
            ]
            return len(self._listeners) < before

    def off_by_id(self, listener_id: str) -> bool:
        """Remove a listener by its ID."""
        with self._lock:
            before = len(self._listeners)
            self._listeners = [l for l in self._listeners if l.listener_id != listener_id]
            return len(self._listeners) < before

    def on_callback(self, event_pattern: str, callback: Callable[[Event], None]) -> None:
        """Register a raw callback (not an agent) for an event."""
        self._callbacks.append((event_pattern, callback))

    # ── Emit ──

    def emit(self, event_name: str, data: dict | None = None, source: str = "") -> EventLog:
        """Fire an event. All matching listeners run their agents in background threads.

        Args:
            event_name: The event name (e.g. "webhook.received", "agent.completed").
            data: Dict of data to pass to listeners (fills query template {placeholders}).
            source: Optional source identifier.

        Returns:
            EventLog with details of what was triggered.
        """
        event = Event(name=event_name, data=data or {}, source=source)
        log = EventLog(event=event)

        # Find matching listeners
        with self._lock:
            matching = [l for l in self._listeners if l.matches(event_name)]

        # Run raw callbacks
        for pattern, cb in self._callbacks:
            if pattern == event_name or fnmatch.fnmatch(event_name, pattern):
                try:
                    cb(event)
                except Exception:
                    pass

        # Run agent listeners in background threads
        for listener in matching:
            log.listeners_triggered += 1
            thread = threading.Thread(
                target=self._execute_listener,
                args=(listener, event, log),
                daemon=True,
            )
            thread.start()

        # Record history (keep last 100)
        with self._lock:
            self._history.append(log)
            if len(self._history) > 100:
                self._history = self._history[-100:]

        return log

    def _execute_listener(self, listener: Listener, event: Event, log: EventLog) -> None:
        """Run a single listener's agent in a thread."""
        query = listener.build_query(event)
        start = time.time()

        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            msg = listener.agent.run(query)

            sys.stdout = old_stdout

            result = msg.content or ""
            cost = sum(e.cost_usd for e in listener.agent.events)
            tokens = sum(e.tokens_used for e in listener.agent.events)

            listener.execution_count += 1
            listener.last_triggered = time.time()

            log.results.append({
                "listener_id": listener.listener_id,
                "agent_name": listener.agent.config.name,
                "query": query,
                "result": result[:500],
                "cost_usd": cost,
                "tokens_used": tokens,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "status": "completed",
            })

        except Exception as e:
            sys.stdout = old_stdout if 'old_stdout' in dir() else sys.__stdout__
            listener.execution_count += 1
            listener.last_triggered = time.time()

            log.results.append({
                "listener_id": listener.listener_id,
                "agent_name": listener.agent.config.name,
                "query": query,
                "result": "",
                "error": str(e),
                "status": "failed",
            })

    # ── Info ──

    def list_listeners(self) -> list[Listener]:
        return list(self._listeners)

    def get_history(self, limit: int = 20) -> list[EventLog]:
        return self._history[-limit:]

    def get_overview(self) -> dict:
        return {
            "total_listeners": len(self._listeners),
            "total_events_emitted": len(self._history),
            "total_executions": sum(l.execution_count for l in self._listeners),
            "event_patterns": list({l.event_pattern for l in self._listeners}),
        }

    def clear(self) -> None:
        """Remove all listeners and history."""
        with self._lock:
            self._listeners.clear()
            self._history.clear()
            self._callbacks.clear()


# ── Module-level singleton ──
event_bus = EventBus()

"""Structured logging with correlation IDs for AgentOS.

Provides JSON-formatted log output with a ``run_id`` that traces a
request through the full execution chain::

    agent.run() → LLM call → tool execution → response

Usage::

    from agentos.logging import get_logger, configure_logging, correlation

    configure_logging()                     # call once at startup
    logger = get_logger("agentos.agent")

    with correlation(agent="my-agent", model="gpt-4o-mini") as ctx:
        logger.info("agent.run.start", extra=ctx(input="hello"))
        # ... ctx carries the same run_id through all nested calls
        logger.info("agent.run.end", extra=ctx(tokens=150, cost=0.001))

Environment variables:
    AGENTOS_LOG_LEVEL   — DEBUG | INFO | WARNING | ERROR (default: INFO)
    AGENTOS_LOG_FORMAT  — json | text (default: json)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Correlation context (thread-safe via contextvars)
# ---------------------------------------------------------------------------

_current_ctx: ContextVar[dict[str, Any]] = ContextVar("agentos_log_ctx", default={})


class CorrelationContext:
    """Holds a ``run_id`` and base attributes for the duration of a run.

    Call the instance like a function to merge extra fields::

        ctx = CorrelationContext(agent="a", model="m")
        logger.info("event", extra=ctx(tokens=100))
    """

    def __init__(self, **base_attrs: Any) -> None:
        self.run_id: str = uuid.uuid4().hex[:12]
        self._base = {"run_id": self.run_id, **base_attrs}

    def __call__(self, **extra: Any) -> dict[str, Any]:
        return {**self._base, **extra}


@contextmanager
def correlation(**base_attrs: Any) -> Generator[CorrelationContext, None, None]:
    """Context manager that sets up a correlation context for the current run.

    All log messages emitted inside the block can include the same
    ``run_id`` by passing ``extra=ctx(...)`` to the logger.
    """
    ctx = CorrelationContext(**base_attrs)
    token = _current_ctx.set(ctx._base)
    try:
        yield ctx
    finally:
        _current_ctx.reset(token)


def get_correlation() -> dict[str, Any]:
    """Return the current correlation context (empty dict if none active)."""
    return dict(_current_ctx.get())


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Outputs one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        for key in ("run_id", "agent", "model", "tool", "step",
                     "tokens", "cost", "latency_ms", "input", "output",
                     "provider", "error", "attempt"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_configured = False


def configure_logging(
    *,
    level: str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure the ``agentos`` logger hierarchy.

    Call once at application startup.  Reads ``AGENTOS_LOG_LEVEL`` and
    ``AGENTOS_LOG_FORMAT`` from the environment if arguments are not
    provided.
    """
    global _configured
    if _configured:
        return
    _configured = True

    log_level = (level or os.getenv("AGENTOS_LOG_LEVEL", "INFO")).upper()
    log_fmt = (fmt or os.getenv("AGENTOS_LOG_FORMAT", "json")).lower()

    root = logging.getLogger("agentos")
    root.setLevel(getattr(logging, log_level, logging.INFO))

    if root.handlers:
        return

    handler = logging.StreamHandler()
    if log_fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-5s [%(name)s] %(message)s "
                "run_id=%(run_id)s"
            )
        )
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str = "agentos") -> logging.Logger:
    """Return a logger under the ``agentos`` namespace."""
    configure_logging()
    return logging.getLogger(name)

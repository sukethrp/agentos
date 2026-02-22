from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime
from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str = ""
    user_id: str = ""
    action_type: str = ""
    resource: str = ""
    outcome: str = ""
    ip_address: str = ""


AUDIT_LOG_PATH = os.environ.get("AGENTOS_AUDIT_LOG", "audit.log")
_lock = asyncio.Lock()


class AuditLogger:
    def __init__(self, log_path: str | None = None):
        self._path = log_path or AUDIT_LOG_PATH

    def _serialize(self, event: AuditEvent) -> str:
        return json.dumps({
            "timestamp": event.timestamp.isoformat(),
            "agent_id": event.agent_id,
            "user_id": event.user_id,
            "action_type": event.action_type,
            "resource": event.resource,
            "outcome": event.outcome,
            "ip_address": event.ip_address,
        }, default=str) + "\n"

    def log_sync(self, event: AuditEvent) -> None:
        with open(self._path, "a") as f:
            f.write(self._serialize(event))

    async def log(self, event: AuditEvent) -> None:
        line = self._serialize(event)

        def _write():
            with open(self._path, "a") as f:
                f.write(line)

        async with _lock:
            await asyncio.to_thread(_write)


_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

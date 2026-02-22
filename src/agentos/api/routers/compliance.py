from __future__ import annotations
import json
import os
from fastapi import APIRouter, Query
from agentos.compliance.audit_logger import AUDIT_LOG_PATH

router = APIRouter(prefix="/compliance", tags=["compliance"])


@router.get("/audit-log")
async def get_audit_log(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    action_type: str = Query("", alias="action_type"),
) -> dict:
    path = os.environ.get("AGENTOS_AUDIT_LOG", AUDIT_LOG_PATH)
    if not os.path.exists(path):
        return {"events": [], "total": 0, "page": page, "limit": limit}
    events: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if action_type and ev.get("action_type", "") != action_type:
                    continue
                events.append(ev)
            except json.JSONDecodeError:
                continue
    total = len(events)
    start = (page - 1) * limit
    end = start + limit
    page_events = events[start:end]
    return {"events": page_events, "total": total, "page": page, "limit": limit}

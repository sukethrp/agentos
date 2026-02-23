from __future__ import annotations
import json
import os
from datetime import datetime
from agentos.compliance.audit_logger import AUDIT_LOG_PATH


def generate_report(
    start: datetime, end: datetime, log_path: str | None = None
) -> dict:
    path = log_path or AUDIT_LOG_PATH
    if not os.path.exists(path):
        return {
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "total_events": 0,
            "by_action_type": {},
            "by_outcome": {},
            "by_agent": {},
            "control_coverage": {"budget": 0, "permission": 0, "audit": 0},
        }
    events: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                ts_str = ev.get("timestamp", "")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if start <= ts <= end:
                        events.append(ev)
            except (json.JSONDecodeError, ValueError):
                continue
    by_action = {}
    by_outcome = {}
    by_agent = {}
    budget_count = 0
    perm_count = 0
    audit_count = 0
    for e in events:
        at = e.get("action_type", "unknown")
        by_action[at] = by_action.get(at, 0) + 1
        out = e.get("outcome", "unknown")
        by_outcome[out] = by_outcome.get(out, 0) + 1
        aid = e.get("agent_id", "unknown")
        by_agent[aid] = by_agent.get(aid, 0) + 1
        if "budget" in at.lower() or "budget" in str(e.get("resource", "")):
            budget_count += 1
        if "permission" in at.lower() or "permission" in str(e.get("resource", "")):
            perm_count += 1
        audit_count += 1
    return {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "total_events": len(events),
        "by_action_type": by_action,
        "by_outcome": by_outcome,
        "by_agent": by_agent,
        "control_coverage": {
            "budget": budget_count,
            "permission": perm_count,
            "audit": audit_count,
        },
    }

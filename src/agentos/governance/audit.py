from __future__ import annotations
import time
import json
from typing import Any


class AuditLog:
    """Immutable audit trail for compliance and debugging.

    Every decision the agent makes is logged with:
    - What happened
    - Why it happened
    - Whether governance allowed or blocked it
    - Full context for investigation
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.entries: list[dict] = []

    def log(
        self,
        action: str,
        details: dict[str, Any] | None = None,
        allowed: bool = True,
        reason: str = "",
        governance_rule: str = "",
    ):
        """Add an audit entry."""
        entry = {
            "timestamp": time.time(),
            "time_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent": self.agent_name,
            "action": action,
            "allowed": allowed,
            "reason": reason,
            "governance_rule": governance_rule,
            "details": details or {},
        }
        self.entries.append(entry)

    def get_entries(self, limit: int = 100) -> list[dict]:
        return self.entries[-limit:]

    def get_blocked(self) -> list[dict]:
        return [e for e in self.entries if not e["allowed"]]

    def get_summary(self) -> dict:
        total = len(self.entries)
        allowed = sum(1 for e in self.entries if e["allowed"])
        blocked = total - allowed
        return {
            "agent": self.agent_name,
            "total_actions": total,
            "allowed": allowed,
            "blocked": blocked,
            "block_rate": f"{(blocked/total*100):.1f}%" if total > 0 else "0%",
        }

    def print_report(self):
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"📋 Audit Report: {self.agent_name}")
        print(f"{'='*60}")
        print(f"   Total actions:  {summary['total_actions']}")
        print(f"   ✅ Allowed:      {summary['allowed']}")
        print(f"   🚫 Blocked:      {summary['blocked']}")
        print(f"   Block rate:     {summary['block_rate']}")
        print(f"{'='*60}")

        blocked = self.get_blocked()
        if blocked:
            print(f"\n   🚫 Blocked Actions:")
            for e in blocked:
                print(f"      [{e['time_readable']}] {e['action']}")
                print(f"         Rule: {e['governance_rule']}")
                print(f"         Reason: {e['reason']}")
                print()

    def export_json(self) -> str:
        return json.dumps(self.entries, indent=2, default=str)
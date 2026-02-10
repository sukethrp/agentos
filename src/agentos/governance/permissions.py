from __future__ import annotations


class PermissionGuard:
    """Controls what tools an agent is allowed to use.

    Usage:
        perms = PermissionGuard(
            allowed_tools=["calculator", "get_weather"],
            blocked_tools=["send_email", "delete_file"],
            require_approval=["company_lookup"],
        )
        ok, msg = perms.check_tool("calculator")
    """

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        blocked_tools: list[str] | None = None,
        require_approval: list[str] | None = None,
        max_actions_per_run: int = 50,
    ):
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.blocked_tools = set(blocked_tools) if blocked_tools else set()
        self.require_approval = set(require_approval) if require_approval else set()
        self.max_actions_per_run = max_actions_per_run

        self.action_count = 0
        self.blocked_count = 0
        self.approval_queue: list[dict] = []

    def check_tool(self, tool_name: str) -> tuple[bool, str]:
        """Check if a tool is allowed. Returns (allowed, message)."""
        self.action_count += 1

        if self.action_count > self.max_actions_per_run:
            self.blocked_count += 1
            return False, f"Max actions per run ({self.max_actions_per_run}) exceeded"

        if tool_name in self.blocked_tools:
            self.blocked_count += 1
            return False, f"Tool '{tool_name}' is blocked by permission policy"

        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            self.blocked_count += 1
            return False, f"Tool '{tool_name}' is not in allowed tools list"

        if tool_name in self.require_approval:
            self.approval_queue.append({"tool": tool_name, "status": "pending"})
            return False, f"Tool '{tool_name}' requires human approval"

        return True, "OK"

    def reset(self):
        """Reset action count for a new run."""
        self.action_count = 0

    def get_status(self) -> dict:
        return {
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools else "all",
            "blocked_tools": list(self.blocked_tools),
            "require_approval": list(self.require_approval),
            "action_count": self.action_count,
            "blocked_count": self.blocked_count,
            "pending_approvals": len([a for a in self.approval_queue if a["status"] == "pending"]),
        }
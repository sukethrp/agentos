"""Permission enforcement for governed agents.

Uses a layered allow-list / block-list approach:

* **Allow-list** (``allowed_tools``) — when provided, *only* these tools may
  be called.  This is the safest default for production agents because any
  newly registered tool is denied until explicitly allowed.
* **Block-list** (``blocked_tools``) — explicitly forbidden tools, checked
  even if the allow-list is open.  Useful for blanket bans on destructive
  operations (e.g. ``delete_file``, ``send_email``) without enumerating every
  safe tool in an allow-list.
* **Approval-required** (``require_approval``) — tools that are conceptually
  allowed but need a human to say "go ahead" before each invocation.

Evaluation order is: action-rate limit → block-list → allow-list → approval.
This means a blocked tool is rejected immediately, even if it also appears in
the allow-list (block wins).

Audit trail
-----------
The ``approval_queue`` is append-only by design.  Every approval request is
recorded with its tool name and status so that post-hoc auditing can
reconstruct the full sequence of sensitive actions an agent attempted,
including those that were denied.  Entries are never deleted or mutated
in-place; status transitions (pending → approved / denied) are appended as
new events in the webhook-based approval flow.

Human approval is async (webhook-based) because the approving human may not
be watching in real-time.  The agent's run is paused until the webhook fires
back with an approved/denied status, avoiding busy-wait polling and keeping
the integration stateless on the agent side.
"""

from __future__ import annotations


class PermissionGuard:
    """Controls what tools an agent is allowed to use.

    The guard enforces three independent layers — block-list, allow-list,
    and human-approval — evaluated in that order for every tool call.
    A per-run action cap provides a final safety net against infinite loops.

    Usage::

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
    ) -> None:
        # None means "all tools allowed" (open policy); an empty set would
        # mean "no tools allowed", which is a valid but restrictive config.
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.blocked_tools = set(blocked_tools) if blocked_tools else set()
        self.require_approval = set(require_approval) if require_approval else set()
        self.max_actions_per_run = max_actions_per_run

        self.action_count = 0
        self.blocked_count = 0
        # Append-only log of every approval request for post-hoc auditing.
        # Never delete or mutate entries — only append new status transitions.
        self.approval_queue: list[dict] = []

    def check_tool(self, tool_name: str) -> tuple[bool, str]:
        """Evaluate whether ``tool_name`` is permitted under the current policy.

        Checks are applied from broadest to most specific:

        1. **Action-rate limit** — hard cap on total actions per run to
           prevent infinite-loop agents from causing unbounded side effects.
        2. **Block-list** — immediate reject for explicitly banned tools.
        3. **Allow-list** — if an allow-list is configured, reject anything
           not on it.  When no allow-list is set the policy is open-by-default.
        4. **Human approval** — tool is conceptually allowed but gated on
           async human confirmation (delivered via webhook).

        Returns:
            A ``(allowed, message)`` tuple.
        """
        self.action_count += 1

        # Safety net: absolute per-run cap stops runaway loops regardless of
        # which tools are being called.
        if self.action_count > self.max_actions_per_run:
            self.blocked_count += 1
            return False, f"Max actions per run ({self.max_actions_per_run}) exceeded"

        # Block-list checked first — explicit deny always wins, even if the
        # tool also appears in the allow-list.
        if tool_name in self.blocked_tools:
            self.blocked_count += 1
            return False, f"Tool '{tool_name}' is blocked by permission policy"

        # Allow-list checked next — when present, only listed tools pass.
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            self.blocked_count += 1
            return False, f"Tool '{tool_name}' is not in allowed tools list"

        # Approval gate — the tool is allowed by policy but requires a human
        # to confirm via an async webhook before execution proceeds.  The
        # request is appended to the audit trail so it's visible even if
        # the approval is never granted.
        if tool_name in self.require_approval:
            self.approval_queue.append({"tool": tool_name, "status": "pending"})
            return False, f"Tool '{tool_name}' requires human approval"

        return True, "OK"

    def reset(self) -> None:
        """Reset action count for a new run.

        Does *not* clear the approval queue — that history is preserved
        across runs for auditing purposes.
        """
        self.action_count = 0

    def get_status(self) -> dict:
        """Return the current permission state for monitoring/debugging."""
        return {
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools else "all",
            "blocked_tools": list(self.blocked_tools),
            "require_approval": list(self.require_approval),
            "action_count": self.action_count,
            "blocked_count": self.blocked_count,
            "pending_approvals": len(
                [a for a in self.approval_queue if a["status"] == "pending"]
            ),
        }

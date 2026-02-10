from __future__ import annotations
from pydantic import BaseModel
from agentos.governance.budget import BudgetGuard
from agentos.governance.permissions import PermissionGuard
from agentos.governance.audit import AuditLog


class GuardrailResult(BaseModel):
    allowed: bool
    message: str
    rule: str = ""


class GovernanceEngine:
    """Central governance for an agent — budget + permissions + audit.

    Usage:
        gov = GovernanceEngine(
            agent_name="my-bot",
            budget=BudgetGuard(max_per_day=5.00),
            permissions=PermissionGuard(blocked_tools=["send_email"]),
        )

        # Before every tool call:
        result = gov.check_tool_call("calculator", estimated_cost=0.001)
        if not result.allowed:
            print(f"BLOCKED: {result.message}")

        # After tool call completes:
        gov.record_action("calculator", cost=0.001, success=True)
    """

    def __init__(
        self,
        agent_name: str,
        budget: BudgetGuard | None = None,
        permissions: PermissionGuard | None = None,
    ):
        self.agent_name = agent_name
        self.budget = budget or BudgetGuard()
        self.permissions = permissions or PermissionGuard()
        self.audit = AuditLog(agent_name)
        self.killed = False
        self.kill_reason = ""

    def check_tool_call(self, tool_name: str, estimated_cost: float = 0.0) -> GuardrailResult:
        """Check if a tool call is allowed by all governance rules."""

        # Check kill switch first
        if self.killed:
            self.audit.log(
                action=f"tool_call:{tool_name}",
                allowed=False,
                reason=f"Agent killed: {self.kill_reason}",
                governance_rule="kill_switch",
            )
            return GuardrailResult(
                allowed=False,
                message=f"🛑 AGENT KILLED: {self.kill_reason}",
                rule="kill_switch",
            )

        # Check permissions
        perm_ok, perm_msg = self.permissions.check_tool(tool_name)
        if not perm_ok:
            self.audit.log(
                action=f"tool_call:{tool_name}",
                allowed=False,
                reason=perm_msg,
                governance_rule="permission",
            )
            return GuardrailResult(allowed=False, message=f"🔒 {perm_msg}", rule="permission")

        # Check budget
        budget_ok, budget_msg = self.budget.check_action(estimated_cost)
        if not budget_ok:
            self.audit.log(
                action=f"tool_call:{tool_name}",
                allowed=False,
                reason=budget_msg,
                governance_rule="budget",
                details={"estimated_cost": estimated_cost},
            )
            return GuardrailResult(allowed=False, message=f"💰 {budget_msg}", rule="budget")

        # All checks passed
        self.audit.log(
            action=f"tool_call:{tool_name}",
            allowed=True,
            reason="All governance checks passed",
            details={"estimated_cost": estimated_cost},
        )
        return GuardrailResult(allowed=True, message="OK")

    def record_action(self, tool_name: str, cost: float, success: bool = True):
        """Record that an action was completed."""
        self.budget.record_spend(cost)
        self.audit.log(
            action=f"completed:{tool_name}",
            allowed=True,
            details={"cost": cost, "success": success},
        )

    def kill(self, reason: str = "Manual kill switch activated"):
        """Emergency stop — immediately halt the agent."""
        self.killed = True
        self.kill_reason = reason
        self.audit.log(
            action="KILL_SWITCH",
            allowed=False,
            reason=reason,
            governance_rule="kill_switch",
        )
        print(f"\n🛑 KILL SWITCH: Agent '{self.agent_name}' has been stopped.")
        print(f"   Reason: {reason}")

    def revive(self):
        """Re-enable the agent after kill switch."""
        self.killed = False
        self.kill_reason = ""
        self.audit.log(action="REVIVE", allowed=True, reason="Agent re-enabled")

    def get_status(self) -> dict:
        return {
            "agent": self.agent_name,
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "budget": self.budget.get_status(),
            "permissions": self.permissions.get_status(),
            "audit_summary": self.audit.get_summary(),
        }

    def print_status(self):
        s = self.get_status()
        status = "🛑 KILLED" if s["killed"] else "✅ Active"
        print(f"\n{'='*60}")
        print(f"🛡️  Governance Status: {self.agent_name}")
        print(f"{'='*60}")
        print(f"   Status:          {status}")
        print(f"   Budget spent:    ${s['budget']['total_spent']:.4f} / ${s['budget']['total_limit']:.2f}")
        print(f"   Budget remaining:${s['budget']['budget_remaining']:.4f}")
        print(f"   Actions:         {s['audit_summary']['total_actions']}")
        print(f"   Blocked:         {s['audit_summary']['blocked']}")
        print(f"{'='*60}")
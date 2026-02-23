from agentos.governance.budget import BudgetGuard
from agentos.governance.permissions import PermissionGuard
from agentos.governance.audit import AuditLog
from agentos.governance.guardrails import GovernanceEngine, GuardrailResult


def test_budgetguard_basic_limits():
    budget = BudgetGuard(
        max_per_action=1.0, max_per_hour=2.0, max_per_day=3.0, max_total=4.0
    )

    ok, msg = budget.check_action(0.5)
    assert ok
    assert msg == "OK"
    budget.record_spend(0.5)
    status = budget.get_status()
    assert status["total_spent"] == 0.5

    ok, msg = budget.check_action(2.0)
    assert not ok
    assert "exceeds per-action limit" in msg


def test_permissionguard_block_and_require_approval():
    perms = PermissionGuard(
        allowed_tools=["calc", "lookup"],
        blocked_tools=["delete"],
        require_approval=["lookup"],
        max_actions_per_run=3,
    )

    ok, msg = perms.check_tool("calc")
    assert ok
    assert msg == "OK"

    ok, msg = perms.check_tool("delete")
    assert not ok
    assert "blocked" in msg

    ok, msg = perms.check_tool("lookup")
    assert not ok
    assert "requires human approval" in msg
    assert perms.approval_queue

    # Exceed max actions per run
    ok, msg = perms.check_tool("calc")
    assert not ok
    assert "Max actions per run" in msg


def test_auditlog_records_and_summaries():
    audit = AuditLog("test-agent")
    audit.log("action1", allowed=True, reason="ok", governance_rule="rule1")
    audit.log("action2", allowed=False, reason="blocked", governance_rule="rule2")

    entries = audit.get_entries()
    assert len(entries) == 2
    blocked = audit.get_blocked()
    assert len(blocked) == 1

    summary = audit.get_summary()
    assert summary["total_actions"] == 2
    assert summary["blocked"] == 1


def test_governanceengine_kill_switch_and_checks():
    gov = GovernanceEngine(agent_name="demo")

    # Initially allowed
    res = gov.check_tool_call("calc", estimated_cost=0.001)
    assert isinstance(res, GuardrailResult)
    assert res.allowed

    # Kill switch
    gov.kill("test reason")
    res2 = gov.check_tool_call("calc", estimated_cost=0.001)
    assert not res2.allowed
    assert res2.rule == "kill_switch"

    status = gov.get_status()
    assert status["killed"] is True
    assert status["kill_reason"] == "test reason"


def test_governanceengine_budget_and_permissions_blocking():
    budget = BudgetGuard(max_per_action=0.1)
    perms = PermissionGuard(blocked_tools=["delete"], allowed_tools=["calc", "read"])
    gov = GovernanceEngine(agent_name="demo2", budget=budget, permissions=perms)

    # Permission block
    res = gov.check_tool_call("delete", estimated_cost=0.001)
    assert not res.allowed
    assert res.rule == "permission"

    # Budget block
    res2 = gov.check_tool_call("calc", estimated_cost=1.0)
    assert not res2.allowed
    assert res2.rule == "budget"

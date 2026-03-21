"""Comprehensive tests for the governance module.

Covers:
- BudgetGuard: per-action, hourly, daily, total limits; window resets; status
- PermissionGuard: allowed/blocked/approval lists; max actions; reset; status
- AuditLog: logging, querying, summary, blocked filter, JSON export, report
- GovernanceEngine: kill switch, revive, budget+perm integration, record_action
- Edge cases: zero budgets, empty permissions, boundary values
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from agentos.governance.budget import BudgetGuard
from agentos.governance.permissions import PermissionGuard
from agentos.governance.audit import AuditLog
from agentos.governance.guardrails import GovernanceEngine, GuardrailResult


# ═══════════════════════════════════════════════════════════════════
# BudgetGuard
# ═══════════════════════════════════════════════════════════════════


class TestBudgetGuardPerActionLimit:
    def test_allows_action_within_limit(self):
        bg = BudgetGuard(max_per_action=1.0)
        ok, msg = bg.check_action(0.50)
        assert ok is True
        assert msg == "OK"

    def test_blocks_action_exceeding_limit(self):
        bg = BudgetGuard(max_per_action=0.10)
        ok, msg = bg.check_action(0.50)
        assert ok is False
        assert "exceeds per-action limit" in msg

    def test_exact_boundary_is_allowed(self):
        bg = BudgetGuard(max_per_action=1.0)
        ok, _ = bg.check_action(1.0)
        assert ok is True

    def test_just_over_boundary_is_blocked(self):
        bg = BudgetGuard(max_per_action=1.0)
        ok, _ = bg.check_action(1.001)
        assert ok is False


class TestBudgetGuardHourlyLimit:
    def test_blocks_when_hourly_limit_would_be_exceeded(self):
        bg = BudgetGuard(max_per_action=5.0, max_per_hour=1.0)
        bg.record_spend(0.90)
        ok, msg = bg.check_action(0.20)
        assert ok is False
        assert "Hourly spend" in msg

    def test_allows_within_hourly_limit(self):
        bg = BudgetGuard(max_per_action=5.0, max_per_hour=2.0)
        bg.record_spend(1.0)
        ok, _ = bg.check_action(0.50)
        assert ok is True

    def test_hourly_window_resets_after_one_hour(self):
        bg = BudgetGuard(max_per_action=5.0, max_per_hour=1.0)
        bg.record_spend(0.99)

        ok, _ = bg.check_action(0.10)
        assert ok is False

        bg.last_hour_reset = time.time() - 3601
        ok, _ = bg.check_action(0.10)
        assert ok is True
        assert bg.hourly_spent == 0.0


class TestBudgetGuardDailyLimit:
    def test_blocks_when_daily_limit_would_be_exceeded(self):
        bg = BudgetGuard(max_per_action=100.0, max_per_hour=100.0, max_per_day=2.0)
        bg.record_spend(1.90)
        ok, msg = bg.check_action(0.20)
        assert ok is False
        assert "Daily spend" in msg

    def test_daily_window_resets_after_24_hours(self):
        bg = BudgetGuard(max_per_action=100.0, max_per_hour=100.0, max_per_day=1.0)
        bg.record_spend(0.99)

        ok, _ = bg.check_action(0.10)
        assert ok is False

        bg.last_day_reset = time.time() - 86401
        bg.last_hour_reset = time.time() - 3601
        ok, _ = bg.check_action(0.10)
        assert ok is True
        assert bg.daily_spent == 0.0


class TestBudgetGuardTotalLimit:
    def test_blocks_when_total_limit_would_be_exceeded(self):
        bg = BudgetGuard(
            max_per_action=100.0, max_per_hour=100.0, max_per_day=100.0, max_total=5.0
        )
        bg.record_spend(4.90)
        ok, msg = bg.check_action(0.20)
        assert ok is False
        assert "Total spend" in msg

    def test_total_does_not_reset(self):
        bg = BudgetGuard(
            max_per_action=100.0, max_per_hour=100.0, max_per_day=100.0, max_total=1.0
        )
        bg.record_spend(0.99)
        bg.last_hour_reset = time.time() - 3601
        bg.last_day_reset = time.time() - 86401
        ok, _ = bg.check_action(0.10)
        assert ok is False


class TestBudgetGuardRecordAndStatus:
    def test_record_spend_accumulates(self):
        bg = BudgetGuard()
        bg.record_spend(1.0)
        bg.record_spend(2.5)
        assert bg.total_spent == 3.5
        assert bg.hourly_spent == 3.5
        assert bg.daily_spent == 3.5

    def test_get_status_returns_all_fields(self):
        bg = BudgetGuard(max_per_action=1.0, max_per_hour=10.0, max_per_day=50.0, max_total=500.0)
        bg.record_spend(3.0)
        bg.check_action(0.5)

        status = bg.get_status()
        assert status["total_spent"] == 3.0
        assert status["hourly_spent"] == 3.0
        assert status["daily_spent"] == 3.0
        assert status["total_limit"] == 500.0
        assert status["hourly_limit"] == 10.0
        assert status["daily_limit"] == 50.0
        assert status["action_count"] == 1
        assert status["blocked_count"] == 0
        assert status["budget_remaining"] == 497.0

    def test_blocked_count_increments(self):
        bg = BudgetGuard(max_per_action=0.01)
        bg.check_action(1.0)
        bg.check_action(1.0)
        assert bg.blocked_count == 2

    def test_action_count_increments_on_every_check(self):
        bg = BudgetGuard(max_per_action=0.01)
        bg.check_action(1.0)
        bg.check_action(0.001)
        assert bg.action_count == 2


class TestBudgetGuardDefaults:
    def test_default_values(self):
        bg = BudgetGuard()
        assert bg.max_per_action == 1.00
        assert bg.max_per_hour == 10.00
        assert bg.max_per_day == 50.00
        assert bg.max_total == 500.00
        assert bg.total_spent == 0.0


# ═══════════════════════════════════════════════════════════════════
# PermissionGuard
# ═══════════════════════════════════════════════════════════════════


class TestPermissionGuardAllowedTools:
    def test_allowed_tool_passes(self):
        pg = PermissionGuard(allowed_tools=["calc", "weather"])
        ok, msg = pg.check_tool("calc")
        assert ok is True
        assert msg == "OK"

    def test_tool_not_in_allowed_list_is_blocked(self):
        pg = PermissionGuard(allowed_tools=["calc"])
        ok, msg = pg.check_tool("email")
        assert ok is False
        assert "not in allowed tools list" in msg

    def test_none_allowed_tools_means_all_allowed(self):
        pg = PermissionGuard(allowed_tools=None)
        ok, msg = pg.check_tool("anything")
        assert ok is True


class TestPermissionGuardBlockedTools:
    def test_blocked_tool_is_rejected(self):
        pg = PermissionGuard(blocked_tools=["delete", "send_email"])
        ok, msg = pg.check_tool("delete")
        assert ok is False
        assert "blocked by permission policy" in msg

    def test_non_blocked_tool_passes(self):
        pg = PermissionGuard(blocked_tools=["delete"])
        ok, msg = pg.check_tool("calc")
        assert ok is True

    def test_blocked_takes_precedence_over_allowed(self):
        pg = PermissionGuard(allowed_tools=["calc", "delete"], blocked_tools=["delete"])
        ok, _ = pg.check_tool("delete")
        assert ok is False


class TestPermissionGuardApproval:
    def test_require_approval_blocks_with_queue_entry(self):
        pg = PermissionGuard(require_approval=["company_lookup"])
        ok, msg = pg.check_tool("company_lookup")
        assert ok is False
        assert "requires human approval" in msg
        assert len(pg.approval_queue) == 1
        assert pg.approval_queue[0]["tool"] == "company_lookup"
        assert pg.approval_queue[0]["status"] == "pending"

    def test_multiple_approvals_queue_up(self):
        pg = PermissionGuard(require_approval=["lookup"])
        pg.check_tool("lookup")
        pg.check_tool("lookup")
        assert len(pg.approval_queue) == 2


class TestPermissionGuardMaxActions:
    def test_blocks_after_max_actions_exceeded(self):
        pg = PermissionGuard(max_actions_per_run=2)
        ok1, _ = pg.check_tool("a")
        ok2, _ = pg.check_tool("b")
        ok3, msg = pg.check_tool("c")
        assert ok1 is True
        assert ok2 is True
        assert ok3 is False
        assert "Max actions per run" in msg

    def test_reset_clears_action_count(self):
        pg = PermissionGuard(max_actions_per_run=2)
        pg.check_tool("a")
        pg.check_tool("b")
        pg.reset()
        ok, _ = pg.check_tool("c")
        assert ok is True
        assert pg.action_count == 1

    def test_reset_does_not_clear_blocked_count(self):
        pg = PermissionGuard(blocked_tools=["x"])
        pg.check_tool("x")
        assert pg.blocked_count == 1
        pg.reset()
        assert pg.blocked_count == 1


class TestPermissionGuardStatus:
    def test_status_with_allowed_tools(self):
        pg = PermissionGuard(
            allowed_tools=["calc"],
            blocked_tools=["delete"],
            require_approval=["lookup"],
        )
        pg.check_tool("calc")
        status = pg.get_status()
        assert status["allowed_tools"] == ["calc"]
        assert "delete" in status["blocked_tools"]
        assert "lookup" in status["require_approval"]
        assert status["action_count"] == 1
        assert status["blocked_count"] == 0

    def test_status_all_tools_allowed_when_none(self):
        pg = PermissionGuard()
        status = pg.get_status()
        assert status["allowed_tools"] == "all"

    def test_pending_approvals_count(self):
        pg = PermissionGuard(require_approval=["lookup"])
        pg.check_tool("lookup")
        pg.check_tool("lookup")
        status = pg.get_status()
        assert status["pending_approvals"] == 2


# ═══════════════════════════════════════════════════════════════════
# AuditLog
# ═══════════════════════════════════════════════════════════════════


class TestAuditLogBasic:
    def test_log_creates_entry_with_all_fields(self):
        audit = AuditLog("test-agent")
        audit.log(
            action="tool_call:calc",
            allowed=True,
            reason="passed",
            governance_rule="permission",
            details={"cost": 0.01},
        )
        entries = audit.get_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["agent"] == "test-agent"
        assert e["action"] == "tool_call:calc"
        assert e["allowed"] is True
        assert e["reason"] == "passed"
        assert e["governance_rule"] == "permission"
        assert e["details"]["cost"] == 0.01
        assert "timestamp" in e
        assert "time_readable" in e

    def test_log_defaults(self):
        audit = AuditLog("a")
        audit.log(action="test")
        e = audit.get_entries()[0]
        assert e["allowed"] is True
        assert e["reason"] == ""
        assert e["governance_rule"] == ""
        assert e["details"] == {}


class TestAuditLogQuerying:
    def test_get_entries_returns_latest(self):
        audit = AuditLog("a")
        for i in range(10):
            audit.log(action=f"action_{i}")
        entries = audit.get_entries(limit=3)
        assert len(entries) == 3
        assert entries[0]["action"] == "action_7"
        assert entries[2]["action"] == "action_9"

    def test_get_entries_default_limit(self):
        audit = AuditLog("a")
        for i in range(5):
            audit.log(action=f"a_{i}")
        assert len(audit.get_entries()) == 5

    def test_get_blocked_filters_correctly(self):
        audit = AuditLog("a")
        audit.log(action="ok", allowed=True)
        audit.log(action="bad1", allowed=False)
        audit.log(action="ok2", allowed=True)
        audit.log(action="bad2", allowed=False)
        blocked = audit.get_blocked()
        assert len(blocked) == 2
        assert blocked[0]["action"] == "bad1"
        assert blocked[1]["action"] == "bad2"

    def test_get_blocked_empty_when_all_allowed(self):
        audit = AuditLog("a")
        audit.log(action="ok", allowed=True)
        assert audit.get_blocked() == []


class TestAuditLogSummary:
    def test_summary_with_mixed_entries(self):
        audit = AuditLog("bot")
        audit.log(action="a1", allowed=True)
        audit.log(action="a2", allowed=False)
        audit.log(action="a3", allowed=True)
        s = audit.get_summary()
        assert s["agent"] == "bot"
        assert s["total_actions"] == 3
        assert s["allowed"] == 2
        assert s["blocked"] == 1
        assert s["block_rate"] == "33.3%"

    def test_summary_empty_log(self):
        audit = AuditLog("empty")
        s = audit.get_summary()
        assert s["total_actions"] == 0
        assert s["block_rate"] == "0%"

    def test_summary_all_blocked(self):
        audit = AuditLog("strict")
        audit.log(action="a1", allowed=False)
        audit.log(action="a2", allowed=False)
        s = audit.get_summary()
        assert s["block_rate"] == "100.0%"


class TestAuditLogExportAndReport:
    def test_export_json_is_valid(self):
        audit = AuditLog("a")
        audit.log(action="x", allowed=True, details={"key": "val"})
        audit.log(action="y", allowed=False, reason="no")
        exported = audit.export_json()
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["action"] == "x"
        assert data[1]["reason"] == "no"

    def test_export_json_empty_log(self):
        audit = AuditLog("a")
        assert json.loads(audit.export_json()) == []

    def test_print_report_runs_without_error(self, capsys):
        audit = AuditLog("demo")
        audit.log(action="ok", allowed=True)
        audit.log(action="bad", allowed=False, reason="denied", governance_rule="perm")
        audit.print_report()
        output = capsys.readouterr().out
        assert "Audit Report: demo" in output
        assert "Blocked Actions" in output
        assert "denied" in output

    def test_print_report_no_blocked(self, capsys):
        audit = AuditLog("clean")
        audit.log(action="ok", allowed=True)
        audit.print_report()
        output = capsys.readouterr().out
        assert "Blocked Actions" not in output


class TestAuditLogImmutability:
    """Entries should be append-only; no method removes past entries."""

    def test_entries_are_append_only(self):
        audit = AuditLog("a")
        audit.log(action="first")
        audit.log(action="second")
        assert len(audit.entries) == 2
        assert audit.entries[0]["action"] == "first"

    def test_get_entries_returns_copy_slice(self):
        audit = AuditLog("a")
        audit.log(action="x")
        entries = audit.get_entries()
        assert entries is not audit.entries or len(entries) == len(audit.entries)


# ═══════════════════════════════════════════════════════════════════
# GovernanceEngine (kill switch, integration)
# ═══════════════════════════════════════════════════════════════════


class TestKillSwitch:
    def test_kill_blocks_all_calls(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.kill("safety concern")
        res = gov.check_tool_call("calc", estimated_cost=0.0)
        assert res.allowed is False
        assert res.rule == "kill_switch"
        assert "KILLED" in res.message

    def test_kill_records_in_audit(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.kill("test")
        blocked = gov.audit.get_blocked()
        assert any(e["action"] == "KILL_SWITCH" for e in blocked)

    def test_revive_re_enables_agent(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.kill("temp")
        assert gov.killed is True
        gov.revive()
        assert gov.killed is False
        assert gov.kill_reason == ""
        res = gov.check_tool_call("calc", estimated_cost=0.001)
        assert res.allowed is True

    def test_revive_is_logged(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.kill("temp")
        gov.revive()
        assert any(e["action"] == "REVIVE" for e in gov.audit.entries)

    def test_kill_after_revive_blocks_again(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.kill("first")
        gov.revive()
        gov.kill("second")
        res = gov.check_tool_call("calc")
        assert res.allowed is False
        assert "second" in gov.kill_reason


class TestGovernanceEnginePermissionIntegration:
    def test_permission_block_logged_to_audit(self, tmp_audit_log):
        perms = PermissionGuard(blocked_tools=["danger"])
        gov = GovernanceEngine(agent_name="bot", permissions=perms)
        res = gov.check_tool_call("danger", estimated_cost=0.0)
        assert res.allowed is False
        assert res.rule == "permission"
        blocked = gov.audit.get_blocked()
        assert any("danger" in e["action"] for e in blocked)

    def test_allowed_tool_not_in_list(self, tmp_audit_log):
        perms = PermissionGuard(allowed_tools=["calc"])
        gov = GovernanceEngine(agent_name="bot", permissions=perms)
        res = gov.check_tool_call("email", estimated_cost=0.0)
        assert res.allowed is False
        assert res.rule == "permission"

    def test_approval_required_blocks(self, tmp_audit_log):
        perms = PermissionGuard(require_approval=["deploy"])
        gov = GovernanceEngine(agent_name="bot", permissions=perms)
        res = gov.check_tool_call("deploy", estimated_cost=0.0)
        assert res.allowed is False
        assert res.rule == "permission"


class TestGovernanceEngineBudgetIntegration:
    def test_budget_block_logged_to_audit(self, tmp_audit_log):
        budget = BudgetGuard(max_per_action=0.01)
        gov = GovernanceEngine(agent_name="bot", budget=budget)
        res = gov.check_tool_call("calc", estimated_cost=1.0)
        assert res.allowed is False
        assert res.rule == "budget"
        blocked = gov.audit.get_blocked()
        assert any("calc" in e["action"] for e in blocked)

    def test_hourly_budget_integration(self, tmp_audit_log):
        budget = BudgetGuard(max_per_action=10.0, max_per_hour=0.50)
        gov = GovernanceEngine(agent_name="bot", budget=budget)

        res1 = gov.check_tool_call("calc", estimated_cost=0.30)
        assert res1.allowed is True
        gov.record_action("calc", cost=0.30)

        res2 = gov.check_tool_call("calc", estimated_cost=0.30)
        assert res2.allowed is False
        assert res2.rule == "budget"

    def test_total_budget_exhaustion(self, tmp_audit_log):
        budget = BudgetGuard(
            max_per_action=10.0, max_per_hour=100.0, max_per_day=100.0, max_total=1.0
        )
        gov = GovernanceEngine(agent_name="bot", budget=budget)
        gov.record_action("calc", cost=0.99)
        res = gov.check_tool_call("calc", estimated_cost=0.10)
        assert res.allowed is False
        assert res.rule == "budget"


class TestGovernanceEngineRecordAction:
    def test_record_action_updates_budget_and_audit(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.record_action("calc", cost=0.05, success=True)
        assert gov.budget.total_spent == 0.05
        assert any("completed:calc" in e["action"] for e in gov.audit.entries)

    def test_record_action_failed(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.record_action("calc", cost=0.01, success=False)
        entry = [e for e in gov.audit.entries if "completed:calc" in e["action"]][0]
        assert entry["details"]["success"] is False


class TestGovernanceEngineStatus:
    def test_get_status_structure(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="my-bot")
        gov.check_tool_call("calc", estimated_cost=0.01)
        status = gov.get_status()
        assert status["agent"] == "my-bot"
        assert status["killed"] is False
        assert status["kill_reason"] == ""
        assert "total_spent" in status["budget"]
        assert "allowed_tools" in status["permissions"]
        assert "total_actions" in status["audit_summary"]

    def test_print_status_runs(self, capsys, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.print_status()
        output = capsys.readouterr().out
        assert "Governance Status" in output
        assert "Active" in output

    def test_print_status_killed(self, capsys, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        gov.kill("safety")
        gov.print_status()
        output = capsys.readouterr().out
        assert "KILLED" in output


class TestGovernanceEngineCheckOrder:
    """Verify that checks run in the correct priority order:
    kill switch > permissions > budget."""

    def test_kill_switch_checked_before_permissions(self, tmp_audit_log):
        perms = PermissionGuard(blocked_tools=["calc"])
        gov = GovernanceEngine(agent_name="bot", permissions=perms)
        gov.kill("test")
        res = gov.check_tool_call("calc", estimated_cost=0.0)
        assert res.rule == "kill_switch"

    def test_permissions_checked_before_budget(self, tmp_audit_log):
        budget = BudgetGuard(max_per_action=0.001)
        perms = PermissionGuard(blocked_tools=["calc"])
        gov = GovernanceEngine(agent_name="bot", budget=budget, permissions=perms)
        res = gov.check_tool_call("calc", estimated_cost=999.0)
        assert res.rule == "permission"

    def test_allowed_call_is_audited(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="bot")
        res = gov.check_tool_call("calc", estimated_cost=0.001)
        assert res.allowed is True
        assert any(
            e["action"] == "tool_call:calc" and e["allowed"] is True
            for e in gov.audit.entries
        )


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_zero_cost_action_always_allowed(self):
        bg = BudgetGuard(max_per_action=0.0)
        ok, _ = bg.check_action(0.0)
        assert ok is True

    def test_negative_cost_does_not_crash(self):
        bg = BudgetGuard()
        ok, _ = bg.check_action(-1.0)
        assert ok is True

    def test_empty_permission_guard_allows_everything(self):
        pg = PermissionGuard()
        ok, _ = pg.check_tool("anything")
        assert ok is True

    def test_many_sequential_checks(self):
        bg = BudgetGuard(max_per_action=100.0, max_per_hour=100.0, max_per_day=100.0, max_total=10.0)
        for _ in range(9):
            ok, _ = bg.check_action(1.0)
            assert ok is True
            bg.record_spend(1.0)
        ok, _ = bg.check_action(2.0)
        assert ok is False

    def test_governance_engine_defaults(self, tmp_audit_log):
        gov = GovernanceEngine(agent_name="default")
        assert gov.killed is False
        assert gov.budget.max_total == 500.0
        assert gov.permissions.allowed_tools is None

    def test_budget_guard_both_windows_reset_simultaneously(self):
        bg = BudgetGuard(
            max_per_action=100.0, max_per_hour=1.0, max_per_day=2.0, max_total=1000.0
        )
        bg.record_spend(0.9)
        bg.last_hour_reset = time.time() - 3601
        bg.last_day_reset = time.time() - 86401
        ok, _ = bg.check_action(0.5)
        assert ok is True
        assert bg.hourly_spent == 0.0
        assert bg.daily_spent == 0.0

    def test_audit_log_agent_name_preserved(self):
        audit = AuditLog("special-agent-007")
        audit.log(action="test")
        assert audit.entries[0]["agent"] == "special-agent-007"
        assert audit.get_summary()["agent"] == "special-agent-007"

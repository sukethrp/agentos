from __future__ import annotations
import asyncio
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch
import pytest

try:
    import anthropic  # noqa: F401 - import for side effect (populates sys.modules)
except ImportError:
    _anthropic = types.ModuleType("anthropic")
    _anthropic.AsyncAnthropic = object
    _anthropic.RateLimitError = Exception
    sys.modules["anthropic"] = _anthropic

_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from agentos.core.types import Message, AgentEvent, Role  # noqa: E402
from agentos.core.tool import Tool  # noqa: E402
from agentos.governance.budget import BudgetGuard  # noqa: E402
from agentos.governance.permissions import PermissionGuard  # noqa: E402
from agentos.governance.audit import AuditLog  # noqa: E402
from agentos.sandbox.scenario import Scenario, ScenarioResult, SandboxReport  # noqa: E402


class MockProvider:
    def __call__(self, *args, **kwargs) -> tuple[Message, AgentEvent]:
        return (
            Message(role=Role.ASSISTANT, content="mock"),
            AgentEvent(
                agent_name="mock",
                event_type="llm_call",
                tokens_used=10,
                cost_usd=0.001,
                latency_ms=50.0,
            ),
        )


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def tmp_audit_log(tmp_path):
    log_path = str(tmp_path / "audit.log")
    with patch.dict(os.environ, {"AGENTOS_AUDIT_LOG": log_path}):
        with patch("agentos.compliance.audit_logger.AUDIT_LOG_PATH", log_path):
            with patch("agentos.compliance.audit_logger._audit_logger", None):
                yield log_path


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Tool fixtures ────────────────────────────────────────────────


@pytest.fixture
def simple_tool():
    """A minimal tool that uppercases a string."""
    def echo(message: str) -> str:
        return message.upper()
    return Tool(echo)


@pytest.fixture
def no_param_tool():
    """A tool that takes no parameters."""
    def get_time() -> str:
        return "2025-01-01T00:00:00Z"
    return Tool(get_time, description="Returns the current time")


# ── Governance fixtures ──────────────────────────────────────────


@pytest.fixture
def strict_budget():
    """A tightly constrained budget for edge-case testing."""
    return BudgetGuard(
        max_per_action=0.05,
        max_per_hour=0.50,
        max_per_day=1.00,
        max_total=5.00,
    )


@pytest.fixture
def open_permissions():
    """A PermissionGuard with no restrictions."""
    return PermissionGuard()


@pytest.fixture
def locked_permissions():
    """A PermissionGuard that only allows 'calculator'."""
    return PermissionGuard(
        allowed_tools=["calculator"],
        blocked_tools=["delete_file", "send_email"],
        require_approval=["deploy"],
    )


@pytest.fixture
def audit_log():
    """A fresh AuditLog instance."""
    return AuditLog("test-agent")


# ── Sandbox fixtures ─────────────────────────────────────────────


@pytest.fixture
def basic_scenario():
    """A minimal valid Scenario for testing."""
    return Scenario(
        name="greeting-test",
        user_message="Hello, how are you?",
        expected_behavior="Respond with a friendly greeting",
    )


@pytest.fixture
def full_scenario():
    """A Scenario with all optional fields populated."""
    return Scenario(
        name="weather-lookup",
        user_message="What's the weather in NYC?",
        expected_behavior="Use weather tool and return a forecast",
        forbidden_actions=["send_email", "delete_file"],
        required_tools=["get_weather"],
        max_cost=0.05,
        max_latency_ms=10000,
        tags=["weather", "tool-use"],
    )


@pytest.fixture
def sample_scenario_result():
    """A pre-built ScenarioResult for report-level tests."""
    return ScenarioResult(
        scenario_name="greeting-test",
        passed=True,
        agent_response="Hello! I'm doing great.",
        relevance_score=9.0,
        safety_score=10.0,
        quality_score=8.0,
        overall_score=9.0,
        cost_usd=0.002,
        latency_ms=150.0,
    )

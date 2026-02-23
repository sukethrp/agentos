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

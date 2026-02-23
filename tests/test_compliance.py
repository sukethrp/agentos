from __future__ import annotations
import pytest
from agentos.compliance.audit_logger import AuditLogger, AuditEvent
from agentos.compliance.data_classifier import DataClassifier
from agentos.compliance.policy_engine import PolicyEngine, PolicyViolationError


@pytest.mark.asyncio
async def test_audit_log_write_and_read(tmp_audit_log):
    import json

    logger = AuditLogger(log_path=tmp_audit_log)
    event = AuditEvent(
        agent_id="agent-1",
        user_id="user-1",
        action_type="query",
        resource="chat",
        outcome="success",
        ip_address="127.0.0.1",
    )
    await logger.log(event)
    with open(tmp_audit_log) as f:
        line = f.readline()
    parsed = json.loads(line)
    assert parsed["agent_id"] == "agent-1"
    assert parsed["user_id"] == "user-1"
    assert parsed["action_type"] == "query"
    assert parsed["resource"] == "chat"
    assert parsed["outcome"] == "success"
    assert parsed["ip_address"] == "127.0.0.1"


def test_data_classifier_phi():
    classifier = DataClassifier()
    result = classifier.classify("Patient SSN: 123-45-6789")
    assert result == "PHI"


def test_data_classifier_public():
    classifier = DataClassifier()
    result = classifier.classify("hello world")
    assert result == "PUBLIC"


def test_policy_engine_phi_violation():
    engine = PolicyEngine()
    with pytest.raises(PolicyViolationError):
        engine.enforce("query", "PHI", "ollama", tool_logged=True)


def test_policy_engine_approved():
    engine = PolicyEngine()
    engine.enforce("query", "PHI", "openai", tool_logged=True)
    engine.enforce("query", "PHI", "anthropic", tool_logged=True)

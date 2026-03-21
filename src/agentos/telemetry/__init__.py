"""OpenTelemetry integration for AgentOS.

Install the optional dependency and call ``setup()`` once at startup::

    pip install 'agentos-platform[otel]'

    from agentos.telemetry import setup
    setup(otlp_endpoint="http://localhost:4317")
"""

from agentos.telemetry.instruments import (
    agent_span,
    llm_span,
    record_llm_metrics,
    record_tool_metrics,
    setup,
    tool_span,
)

__all__ = [
    "agent_span",
    "llm_span",
    "record_llm_metrics",
    "record_tool_metrics",
    "setup",
    "tool_span",
]

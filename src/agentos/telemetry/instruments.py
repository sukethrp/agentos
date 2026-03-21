"""OpenTelemetry instruments for AgentOS.

All functions are safe to call even when OpenTelemetry is not installed —
they degrade to no-ops.  Install the ``otel`` extra to enable:

    pip install 'agentos-platform[otel]'
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from agentos.core.types import AgentEvent

# ---------------------------------------------------------------------------
# Lazy OpenTelemetry initialisation
# ---------------------------------------------------------------------------

_HAS_OTEL = False
_tracer: Any = None
_meter: Any = None

_llm_token_counter: Any = None
_llm_cost_counter: Any = None
_llm_latency_hist: Any = None
_tool_latency_hist: Any = None
_agent_run_counter: Any = None

_ATTR_AGENT = "agentos.agent.name"
_ATTR_MODEL = "agentos.llm.model"
_ATTR_TOOL = "agentos.tool.name"


def _try_init() -> None:
    global _HAS_OTEL, _tracer, _meter  # noqa: PLW0603
    global _llm_token_counter, _llm_cost_counter, _llm_latency_hist  # noqa: PLW0603
    global _tool_latency_hist, _agent_run_counter  # noqa: PLW0603
    if _tracer is not None:
        return
    try:
        from opentelemetry import metrics, trace

        _tracer = trace.get_tracer("agentos")
        _meter = metrics.get_meter("agentos")

        _llm_token_counter = _meter.create_counter(
            "agentos.llm.tokens",
            unit="token",
            description="Total LLM tokens consumed",
        )
        _llm_cost_counter = _meter.create_counter(
            "agentos.llm.cost",
            unit="USD",
            description="Total LLM cost in USD",
        )
        _llm_latency_hist = _meter.create_histogram(
            "agentos.llm.latency",
            unit="ms",
            description="LLM call latency",
        )
        _tool_latency_hist = _meter.create_histogram(
            "agentos.tool.latency",
            unit="ms",
            description="Tool execution latency",
        )
        _agent_run_counter = _meter.create_counter(
            "agentos.agent.runs",
            description="Total agent run invocations",
        )
        _HAS_OTEL = True
    except Exception:
        _HAS_OTEL = False


# ---------------------------------------------------------------------------
# Setup — call once at application startup
# ---------------------------------------------------------------------------


def setup(
    *,
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    otlp_protocol: str | None = None,
    console_export: bool = False,
) -> None:
    """Configure the OpenTelemetry SDK with an OTLP exporter.

    Parameters are read from arguments first, then standard OTel env vars
    (``OTEL_SERVICE_NAME``, ``OTEL_EXPORTER_OTLP_ENDPOINT``, etc.).

    Call this **before** any agent runs so that spans and metrics are
    exported.
    """
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The 'opentelemetry' packages are required. "
            "Install with: pip install 'agentos-platform[otel]'"
        ) from None

    svc = service_name or os.getenv("OTEL_SERVICE_NAME", "agentos")
    resource = Resource.create({"service.name": svc})

    # ── Traces ──────────────────────────────────────────────────────
    tp = TracerProvider(resource=resource)

    if console_export:
        tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    protocol = otlp_protocol or os.getenv(
        "OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"
    )
    if endpoint:
        if protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        tp.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )

    trace.set_tracer_provider(tp)

    # ── Metrics ─────────────────────────────────────────────────────
    readers = []
    if console_export:
        readers.append(
            PeriodicExportingMetricReader(ConsoleMetricExporter())
        )
    if endpoint:
        if protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=endpoint)
            )
        )

    mp = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(mp)

    _try_init()


# ---------------------------------------------------------------------------
# Span context managers — safe to call without OTel installed
# ---------------------------------------------------------------------------


@contextmanager
def agent_span(
    agent_name: str, model: str, user_input: str
) -> Generator[Any, None, None]:
    """Root span wrapping an entire ``agent.run()`` call."""
    _try_init()
    if not _HAS_OTEL:
        yield None
        return

    with _tracer.start_as_current_span("agent.run") as span:
        span.set_attribute(_ATTR_AGENT, agent_name)
        span.set_attribute(_ATTR_MODEL, model)
        span.set_attribute("agentos.agent.input", user_input[:500])
        _agent_run_counter.add(1, {_ATTR_AGENT: agent_name, _ATTR_MODEL: model})
        yield span


@contextmanager
def llm_span(model: str) -> Generator[Any, None, None]:
    """Child span wrapping a single LLM provider call."""
    _try_init()
    if not _HAS_OTEL:
        yield None
        return

    with _tracer.start_as_current_span("llm.call") as span:
        span.set_attribute(_ATTR_MODEL, model)
        yield span


@contextmanager
def tool_span(tool_name: str) -> Generator[Any, None, None]:
    """Child span wrapping a single tool execution."""
    _try_init()
    if not _HAS_OTEL:
        yield None
        return

    with _tracer.start_as_current_span("tool.execute") as span:
        span.set_attribute(_ATTR_TOOL, tool_name)
        yield span


# ---------------------------------------------------------------------------
# Metric recording helpers
# ---------------------------------------------------------------------------


def record_llm_metrics(event: AgentEvent) -> None:
    """Record token, cost, and latency metrics from an LLM AgentEvent."""
    _try_init()
    if not _HAS_OTEL:
        return

    attrs = {
        _ATTR_AGENT: event.agent_name,
        _ATTR_MODEL: event.data.get("model", "unknown"),
    }
    _llm_token_counter.add(event.tokens_used, attrs)
    _llm_cost_counter.add(event.cost_usd, attrs)
    _llm_latency_hist.record(event.latency_ms, attrs)


def record_tool_metrics(
    agent_name: str, tool_name: str, latency_ms: float
) -> None:
    """Record latency metrics for a tool execution."""
    _try_init()
    if not _HAS_OTEL:
        return

    attrs = {_ATTR_AGENT: agent_name, _ATTR_TOOL: tool_name}
    _tool_latency_hist.record(latency_ms, attrs)

from __future__ import annotations

from agentos.observability.alerts import AlertEngine
from agentos.observability.run_viewer import build_run_view
from agentos.observability.tracer import (
    StepType,
    Trace,
    TraceBuilder,
    TraceStep,
    TraceStore,
)


def test_tracer_span():
    builder = TraceBuilder(agent_name="test", model="gpt-4o-mini", system_prompt="Hi")
    builder.set_query("hello")
    builder.add_llm_call([], [], tokens=10, cost=0.001, latency_ms=100.0)
    builder.add_final_answer("hi")
    trace = builder.finish()
    assert trace.agent_name == "test"
    assert trace.total_latency_ms > 0
    assert len(trace.steps) >= 2


def test_alert_threshold():
    store = TraceStore(max_traces=100)
    for i in range(5):
        t = Trace(agent_name="a", trace_id=f"t{i}", success=True)
        t.steps.append(
            TraceStep(
                step_type=StepType.TOOL_CALL,
                tool_name="web_search",
                is_error=(i < 4),
            )
        )
        store.add(t)
    engine = AlertEngine(store=store)
    alerts = engine.evaluate()
    assert len(alerts) >= 1
    tool_alerts = [
        a for a in alerts if "web_search" in a.title or "web_search" in a.cause
    ]
    assert len(tool_alerts) >= 1


def test_run_view_renders_a_frame_per_step():
    builder = TraceBuilder(
        agent_name="run-view-agent", model="gpt-4o", system_prompt="Help"
    )
    builder.set_query("test query")
    builder.add_llm_call(
        [{"role": "user", "content": "test"}], ["tool1"], tokens=5, latency_ms=50.0
    )
    builder.add_tool_call("tool1", {"q": "x"}, result="ok", latency_ms=10.0)
    builder.add_final_answer("done")
    trace = builder.finish()
    view = build_run_view(trace, include_messages=False)
    assert view.trace_id == trace.trace_id
    assert view.agent_name == trace.agent_name
    assert view.user_query == trace.user_query
    assert len(view.frames) >= 4
    step_labels = [f.label for f in view.frames]
    assert any("LLM" in lb or "SETUP" in lb for lb in step_labels)
    assert any("TOOL" in lb for lb in step_labels)
    assert any("ANSWER" in lb or "OUTCOME" in lb for lb in step_labels)


def test_step_to_frame_warn_when_diagnosis_is_warn_not_fail():
    """Amber path: root-cause step at WARN severity must not collapse to ok/fail."""
    from agentos.observability.diagnostics import Diagnosis, Severity
    from agentos.observability.run_viewer import _step_to_frame

    step = TraceStep(
        step_type=StepType.LLM_CALL,
        step_index=2,
        is_error=False,
        tokens_used=10,
        cost_usd=0.0,
        latency_ms=5.0,
        messages_snapshot=[],
        available_tools=[],
    )
    diag = Diagnosis(
        overall_severity=Severity.WARN,
        root_cause_step=2,
        root_cause="soft warning",
    )
    frame = _step_to_frame(step, frame_idx=0, diag=diag, include_messages=False)
    assert frame.severity == "warn"


def test_step_to_frame_fail_when_step_errors():
    from agentos.observability.diagnostics import Diagnosis, Severity
    from agentos.observability.run_viewer import _step_to_frame

    step = TraceStep(
        step_type=StepType.TOOL_CALL,
        step_index=1,
        is_error=True,
        tool_name="x",
        tool_arguments={},
        tool_result="err",
    )
    diag = Diagnosis(overall_severity=Severity.PASS)
    frame = _step_to_frame(step, frame_idx=0, diag=diag, include_messages=False)
    assert frame.severity == "fail"


def test_run_view_text_includes_severity_icons():
    builder = TraceBuilder(agent_name="icons", model="gpt-4o", system_prompt="Help")
    builder.set_query("q")
    builder.add_llm_call([], [], tokens=1, latency_ms=1.0)
    builder.add_final_answer("ok")
    view = build_run_view(builder.finish(), include_messages=False)
    text = view.text()
    assert "·" in text or "✗" in text or "⚠" in text

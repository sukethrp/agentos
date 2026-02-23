from __future__ import annotations
from agentos.observability.tracer import (
    TraceBuilder,
    TraceStore,
    Trace,
    TraceStep,
    StepType,
)
from agentos.observability.alerts import AlertEngine
from agentos.observability.replay import build_replay


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


def test_replay_records_and_replays():
    builder = TraceBuilder(
        agent_name="replay-agent", model="gpt-4o", system_prompt="Help"
    )
    builder.set_query("test query")
    builder.add_llm_call(
        [{"role": "user", "content": "test"}], ["tool1"], tokens=5, latency_ms=50.0
    )
    builder.add_tool_call("tool1", {"q": "x"}, result="ok", latency_ms=10.0)
    builder.add_final_answer("done")
    trace = builder.finish()
    replay = build_replay(trace, include_messages=False)
    assert replay.trace_id == trace.trace_id
    assert replay.agent_name == trace.agent_name
    assert replay.user_query == trace.user_query
    assert len(replay.frames) >= 4
    step_labels = [f.label for f in replay.frames]
    assert any("LLM" in lb or "SETUP" in lb for lb in step_labels)
    assert any("TOOL" in lb for lb in step_labels)
    assert any("ANSWER" in lb or "OUTCOME" in lb for lb in step_labels)

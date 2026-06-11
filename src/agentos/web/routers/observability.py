from __future__ import annotations
from fastapi import APIRouter
from agentos.observability.tracer import (
    get_trace_store as _obs_trace_store,
    TraceBuilder,
)
from agentos.observability.diagnostics import diagnose as _obs_diagnose
from agentos.observability.alerts import AlertEngine as _ObsAlertEngine
from agentos.observability.replay import build_replay as _obs_build_replay

router = APIRouter(tags=["observability"])

@router.get("/api/observability/stats")
async def obs_stats():
    ts = _obs_trace_store()
    return ts.stats()


@router.get("/api/observability/traces")
async def obs_traces(limit: int = 20, agent: str = ""):
    ts = _obs_trace_store()
    traces = ts.list_all(agent_name=agent, limit=limit)
    return [t.to_dict() for t in traces]


@router.get("/api/observability/trace/{trace_id}")
async def obs_trace_detail(trace_id: str):
    ts = _obs_trace_store()
    t = ts.get(trace_id)
    if not t:
        return {"error": "Trace not found"}
    return t.to_dict()


@router.get("/api/observability/diagnose/{trace_id}")
async def obs_diagnose_trace(trace_id: str):
    ts = _obs_trace_store()
    t = ts.get(trace_id)
    if not t:
        return {"error": "Trace not found"}
    diag = _obs_diagnose(t)
    return diag.to_dict()


@router.get("/api/observability/alerts")
async def obs_alerts(agent: str = ""):
    ts = _obs_trace_store()
    engine = _ObsAlertEngine(ts)
    alerts = engine.evaluate(agent_name=agent)
    return [a.to_dict() for a in alerts]


@router.get("/api/observability/replay/{trace_id}")
async def obs_replay(trace_id: str):
    ts = _obs_trace_store()
    t = ts.get(trace_id)
    if not t:
        return {"error": "Trace not found"}
    replay = _obs_build_replay(t, include_messages=True)
    return replay.to_dict()


@router.post("/api/observability/seed-demo")
async def obs_seed_demo():
    """Seed example traces for demonstration."""
    ts = _obs_trace_store()
    b = TraceBuilder(
        "demo-agent", "gpt-4o-mini", "You are a helpful assistant with tools."
    )
    b.set_query("What's the weather in Tokyo?")
    b.add_llm_call(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        ["weather"],
        150,
        0.0003,
        450,
    )
    b.add_tool_call(
        "weather",
        {"location": "Tokyo"},
        '{"temperature": "22C", "condition": "Sunny"}',
        120,
    )
    b.add_llm_call(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "tool", "content": '{"temperature": "22C"}'},
        ],
        ["weather"],
        180,
        0.0004,
        380,
    )
    b.add_final_answer("It is currently 22C and sunny in Tokyo.")
    ts.add(b.finish())
    b2 = TraceBuilder(
        "demo-agent", "gpt-4o-mini", "You are a helpful assistant with tools."
    )
    b2.set_query("What's the weather in Paris?")
    b2.add_llm_call(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ],
        ["weather"],
        140,
        0.0003,
        420,
    )
    b2.add_tool_call(
        "weather", {"location": "Paris"}, "ERROR: API rate limit exceeded", 2100
    )
    b2.add_error("Tool returned error — unreliable response")
    ts.add(b2.finish())
    b3 = TraceBuilder("support-bot", "gpt-4o-mini", "You are a customer support agent.")
    b3.set_query("Check order #12345")
    b3.add_llm_call(
        [
            {"role": "system", "content": "Support agent."},
            {"role": "user", "content": "Check order #12345"},
        ],
        ["search"],
        120,
        0.0002,
        350,
    )
    b3.add_tool_call(
        "order_lookup",
        {"order_id": "12345"},
        "ERROR: Tool 'order_lookup' not found",
        1,
        not_found=True,
    )
    b3.add_error("LLM hallucinated tool name 'order_lookup'")
    ts.add(b3.finish())
    return {"seeded": 3, "total": ts.stats()["total_traces"]}


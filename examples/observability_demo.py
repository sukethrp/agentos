#!/usr/bin/env python3
"""
Observability Demo — deep tracing, root cause analysis, and step-by-step replay.

This demo constructs synthetic traces (no API calls) to show:
1. A healthy interaction (all checks pass)
2. A tool error failure (tool returns an error → LLM hallucinates)
3. A missing tool failure (LLM calls a tool that doesn't exist)
4. A context problem (empty system prompt)
5. Smart alerts that explain *why* quality dropped
6. Step-by-step replay of the failed interaction

Run:
    python examples/observability_demo.py
"""

from __future__ import annotations

import json
import textwrap

from agentos.observability.tracer import (
    Trace,
    TraceBuilder,
    TraceStore,
    StepType,
)
from agentos.observability.diagnostics import diagnose, diagnose_batch
from agentos.observability.alerts import AlertEngine
from agentos.observability.replay import build_replay


DIVIDER = "═" * 60


def pp(label: str, obj: dict | list | str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(DIVIDER)
    if isinstance(obj, (dict, list)):
        print(textwrap.indent(json.dumps(obj, indent=2, default=str)[:1500], "  "))
    else:
        print(textwrap.indent(str(obj)[:1500], "  "))


def _build_healthy_trace() -> Trace:
    """Trace of a successful interaction: user asks for weather → tool called → good answer."""
    b = TraceBuilder(
        agent_name="weather-bot",
        model="gpt-4o-mini",
        system_prompt="You are a helpful weather assistant. Use the weather tool for accurate data.",
    )
    b.set_query("What's the weather in Tokyo?")
    b.add_llm_call(
        messages=[
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        available_tools=["weather"],
        tokens=150,
        cost=0.0003,
        latency_ms=450,
    )
    b.add_tool_call(
        tool_name="weather",
        arguments={"location": "Tokyo"},
        result='{"temperature": "22°C", "condition": "Partly Cloudy", "humidity": "65%"}',
        latency_ms=120,
    )
    b.add_llm_call(
        messages=[
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "weather"}}]},
            {"role": "tool", "tool_call_id": "1", "content": '{"temperature": "22°C", "condition": "Partly Cloudy"}'},
        ],
        available_tools=["weather"],
        tokens=180,
        cost=0.0004,
        latency_ms=380,
    )
    b.add_final_answer("The weather in Tokyo is currently 22°C and partly cloudy with 65% humidity.")
    return b.finish()


def _build_tool_error_trace() -> Trace:
    """Trace where the tool returns an error → LLM is forced to guess."""
    b = TraceBuilder(
        agent_name="weather-bot",
        model="gpt-4o-mini",
        system_prompt="You are a helpful weather assistant. Use the weather tool for accurate data.",
    )
    b.set_query("What's the weather in Paris?")
    b.add_llm_call(
        messages=[
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ],
        available_tools=["weather"],
        tokens=140,
        cost=0.0003,
        latency_ms=420,
    )
    b.add_tool_call(
        tool_name="weather",
        arguments={"location": "Paris"},
        result="ERROR: API rate limit exceeded. Try again later.",
        latency_ms=2100,
    )
    b.add_llm_call(
        messages=[
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "tool", "tool_call_id": "1", "content": "ERROR: API rate limit exceeded."},
        ],
        available_tools=["weather"],
        tokens=200,
        cost=0.0004,
        latency_ms=500,
    )
    b.add_final_answer("Based on typical conditions, Paris is probably around 15°C today.")
    trace = b.finish()
    trace.success = False
    trace.error = "Tool error caused unreliable response"
    return trace


def _build_missing_tool_trace() -> Trace:
    """Trace where the LLM calls a tool that doesn't exist."""
    b = TraceBuilder(
        agent_name="support-bot",
        model="gpt-4o-mini",
        system_prompt="You are a customer support assistant.",
    )
    b.set_query("Check my order status for order #12345")
    b.add_llm_call(
        messages=[
            {"role": "system", "content": "You are a customer support assistant."},
            {"role": "user", "content": "Check my order status for order #12345"},
        ],
        available_tools=["web_search", "calculator"],
        tokens=120,
        cost=0.0002,
        latency_ms=350,
    )
    b.add_tool_call(
        tool_name="order_lookup",
        arguments={"order_id": "12345"},
        result="ERROR: Tool 'order_lookup' not found",
        not_found=True,
        latency_ms=1,
    )
    b.add_error("Tool 'order_lookup' does not exist — LLM hallucinated the tool name")
    trace = b.finish()
    return trace


def _build_no_prompt_trace() -> Trace:
    """Trace with an empty system prompt."""
    b = TraceBuilder(
        agent_name="generic-bot",
        model="gpt-4o-mini",
        system_prompt="",
    )
    b.set_query("Explain quantum computing")
    b.add_llm_call(
        messages=[{"role": "user", "content": "Explain quantum computing"}],
        available_tools=[],
        tokens=300,
        cost=0.0006,
        latency_ms=800,
    )
    b.add_final_answer("Quantum computing uses qubits.")
    return b.finish()


def main() -> None:
    print("🔍 AgentOS Observability Demo — Root Cause Analysis")
    print("=" * 60)

    store = TraceStore()

    # ── 1. Build traces ──────────────────────────────────────────────────

    print("\n📋 Building synthetic traces…")

    healthy = _build_healthy_trace()
    tool_error = _build_tool_error_trace()
    missing_tool = _build_missing_tool_trace()
    no_prompt = _build_no_prompt_trace()

    traces = [healthy, tool_error, missing_tool, no_prompt]
    for t in traces:
        store.add(t)
    # Add extra copies of tool_error to trigger alerts
    for _ in range(4):
        extra = _build_tool_error_trace()
        store.add(extra)

    print(f"  Created {len(store.list_all())} traces")

    # ── 2. Trace timeline ────────────────────────────────────────────────

    print("\n\n📊 STEP 1: Trace Timelines")
    print("-" * 40)
    for t in traces:
        print(f"\n{t.timeline()}")

    # ── 3. Diagnostics — root cause analysis ─────────────────────────────

    print("\n\n🩺 STEP 2: Root Cause Diagnostics")
    print("-" * 40)

    for t in traces:
        diag = diagnose(t)
        print(diag.summary_text())

    # ── 4. Batch diagnostics ─────────────────────────────────────────────

    print("\n\n📋 STEP 3: Batch Diagnosis (sorted by severity)")
    print("-" * 40)

    all_diags = diagnose_batch(traces)
    for d in all_diags:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(d.overall_severity.value, "?")
        print(f"  {icon} [{d.trace_id}] {d.agent_name}: {d.root_cause}")

    # ── 5. Smart alerts ──────────────────────────────────────────────────

    print("\n\n🚨 STEP 4: Smart Alerts")
    print("-" * 40)

    engine = AlertEngine(store)
    alerts = engine.evaluate()

    if not alerts:
        print("  No alerts generated.")
    for a in alerts:
        print(f"\n  {a.summary()}")
        print(f"    Cause: {a.cause}")
        print(f"    Impact: {a.impact}")
        print(f"    Fix: {a.recommendation}")

    # ── 6. Replay a failed interaction ───────────────────────────────────

    print("\n\n🔄 STEP 5: Step-by-Step Replay (tool error)")
    print("-" * 40)

    replay = build_replay(tool_error, include_messages=True)
    print(replay.text())

    print("\n\n🔄 STEP 6: Step-by-Step Replay (missing tool)")
    print("-" * 40)

    replay2 = build_replay(missing_tool, include_messages=True)
    print(replay2.text())

    # ── Summary ──────────────────────────────────────────────────────────

    pp("Trace Store Stats", store.stats())

    print(f"\n{'=' * 60}")
    print("✅ Observability Demo Complete!")
    print(f"   Traces: {len(store.list_all())}")
    print(f"   Diagnoses: {len(all_diags)}")
    print(f"   Alerts: {len(alerts)}")
    print(f"   5-point diagnostic checks: context, tool selection,")
    print(f"     tool execution, interpretation, faithfulness")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""AgentOS Event Bus Demo — agents reacting to events in real-time.

Shows:
  1. Agent-to-agent chaining (researcher completes → analyst reacts)
  2. Webhook event handling
  3. Timer-triggered events
  4. Custom event emission
"""

import sys
import time

sys.path.insert(0, "src")

from agentos.core.agent import Agent
from agentos.tools import get_builtin_tools
from agentos.events import (
    event_bus,
    WebhookTrigger,
    TimerTrigger,
    AgentCompleteTrigger,
)


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    divider("⚡ AgentOS — Event Bus Demo")

    tools = list(get_builtin_tools().values())

    # ── Create agents ──
    researcher = Agent(
        name="researcher",
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="You are a research assistant. Give concise 1-2 sentence answers.",
    )

    analyst = Agent(
        name="analyst",
        model="gpt-4o-mini",
        tools=[],
        system_prompt="You are a data analyst. Summarize findings in 1-2 bullet points. Be very concise.",
    )

    webhook_agent = Agent(
        name="webhook-handler",
        model="gpt-4o-mini",
        tools=[],
        system_prompt="You process webhook payloads. Describe what happened in one sentence.",
    )

    timer_agent = Agent(
        name="timer-bot",
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="You respond to timer events. Give a one-line status update.",
    )

    # ══════════════════════════════════════════════════════
    #  Demo 1: Agent-to-Agent Chaining
    # ══════════════════════════════════════════════════════
    divider("Demo 1: Agent-to-Agent Chaining")

    # When researcher completes, the analyst should analyze the result
    event_bus.on(
        "agent.completed",
        analyst,
        "The researcher just completed a task. Their findings: {result}\n\nPlease analyze and summarize the key points.",
    )

    # Set up the AgentCompleteTrigger so researcher fires "agent.completed" after run
    agent_trigger = AgentCompleteTrigger(
        name="researcher-done",
        watched_agent_name="researcher",
    )
    agent_trigger.start()

    # Wrap the researcher so it auto-fires the trigger on completion
    wrapped_researcher = agent_trigger.wrap_agent(researcher)

    print("📋 Registered: analyst listens for 'agent.completed'")
    print("🔗 Wrapped: researcher fires 'agent.completed' after each run")
    print()

    # Run the researcher — the analyst should react automatically
    print("▶️  Running researcher: 'What is the capital of France?'")
    result = wrapped_researcher.run("What is the capital of France and what is its population?")
    print(f"   Researcher says: {result.content[:150]}")

    # Give the analyst time to react
    print("\n⏳ Waiting for analyst to react...")
    time.sleep(8)

    # Check event history
    history = event_bus.get_history(limit=5)
    for log in history:
        print(f"   📨 Event '{log.event.name}' → {log.listeners_triggered} listener(s)")
        for r in log.results:
            status = "✅" if r.get("status") == "completed" else "❌"
            print(f"      {status} {r.get('agent_name', '?')}: {r.get('result', '')[:100]}")

    # ══════════════════════════════════════════════════════
    #  Demo 2: Webhook Events
    # ══════════════════════════════════════════════════════
    divider("Demo 2: Webhook Events")

    # Register webhook handler
    event_bus.on(
        "webhook.*",
        webhook_agent,
        "A webhook was received: event={event_name}, data={data}. Describe what happened.",
    )

    print("📋 Registered: webhook-handler listens for 'webhook.*'")
    print()

    # Simulate a webhook
    print("🔔 Simulating webhook: deployment completed")
    log = event_bus.emit(
        "webhook.deploy",
        data={
            "repo": "agentos",
            "branch": "main",
            "status": "success",
            "commit": "abc123",
            "data": '{"repo":"agentos","status":"success"}',
        },
        source="github-actions",
    )
    print(f"   Emitted → {log.listeners_triggered} listener(s) triggered")

    time.sleep(6)

    # Show results
    recent = event_bus.get_history(limit=1)
    if recent:
        for r in recent[-1].results:
            status = "✅" if r.get("status") == "completed" else "❌"
            print(f"   {status} {r.get('agent_name', '?')}: {r.get('result', '')[:120]}")

    # ══════════════════════════════════════════════════════
    #  Demo 3: Timer Events
    # ══════════════════════════════════════════════════════
    divider("Demo 3: Timer Events (2 ticks)")

    # Register timer handler
    event_bus.on(
        "timer.fired",
        timer_agent,
        "Timer tick #{fire_count}. Give a one-line status check.",
    )

    print("📋 Registered: timer-bot listens for 'timer.fired'")

    # Create and start a timer
    timer = TimerTrigger(
        name="status-timer",
        interval_seconds=5,
        event_name="timer.fired",
        max_fires=2,
    )
    timer.start()
    print("⏰ Timer started: fires every 5 seconds, max 2 times")
    print("⏳ Waiting for timer events...")

    # Wait for timer to fire twice (2 x 5s + buffer)
    time.sleep(14)
    timer.stop()

    # Show timer results
    recent = event_bus.get_history(limit=3)
    for log in recent:
        if "timer" in log.event.name:
            print(f"   ⏰ Timer tick → {log.listeners_triggered} listener(s)")
            for r in log.results:
                status = "✅" if r.get("status") == "completed" else "❌"
                print(f"      {status} {r.get('agent_name', '?')}: {r.get('result', '')[:100]}")

    # ══════════════════════════════════════════════════════
    #  Demo 4: Custom Events
    # ══════════════════════════════════════════════════════
    divider("Demo 4: Custom Events")

    custom_results = []

    # Use a raw callback for fast, non-agent reactions
    def on_custom(event):
        custom_results.append(event)
        ts = time.strftime("%H:%M:%S")
        print(f"   ⚡ [{ts}] Custom callback received: {event.name} → {event.data}")

    event_bus.on_callback("custom.*", on_custom)
    print("📋 Registered: raw callback for 'custom.*'")
    print()

    # Emit a few custom events
    for i in range(3):
        event_bus.emit(f"custom.event_{i+1}", data={"index": i + 1, "data": f"payload-{i+1}"})
        time.sleep(0.5)

    print(f"\n   Total custom events caught: {len(custom_results)}")

    # ══════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════
    divider("Summary")

    overview = event_bus.get_overview()
    print(f"   Total listeners:      {overview['total_listeners']}")
    print(f"   Total events emitted: {overview['total_events_emitted']}")
    print(f"   Total executions:     {overview['total_executions']}")
    print(f"   Event patterns:       {overview['event_patterns']}")

    print()
    print("   Registered listeners:")
    for l in event_bus.list_listeners():
        agent_name = getattr(l.agent, "config", None) and l.agent.config.name or "callback"
        print(f"     • {l.event_pattern:25s} → {agent_name} (ran {l.execution_count}x)")

    # Clean up
    event_bus.clear()
    print(f"\n{'=' * 60}")
    print("  ✅ Demo complete — event bus cleared")
    print(f"{'=' * 60}")

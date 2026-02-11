"""AgentOS Scheduler Demo — run agents on automated schedules."""

import sys
import time

sys.path.insert(0, "src")

from agentos.scheduler import AgentScheduler
from agentos.core.agent import Agent
from agentos.tools import get_builtin_tools


if __name__ == "__main__":
    print("=" * 60)
    print("⏰ AgentOS — Scheduler Demo")
    print("=" * 60)

    tools = list(get_builtin_tools().values())

    # Create agents
    weather_agent = Agent(
        name="weather-bot",
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="You are a weather reporter. Give brief weather updates. Be concise — one sentence max.",
    )

    report_agent = Agent(
        name="report-bot",
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="You are a quick calculator. Solve the math and give just the answer.",
    )

    # Create scheduler
    scheduler = AgentScheduler(max_concurrent=2)

    # Register a callback to see executions live
    def on_exec(job, execution):
        status = "✅" if execution.status == "completed" else "❌"
        ts = time.strftime("%H:%M:%S")
        print(f"\n{status} [{ts}] Job '{job.agent_name}' (run #{job.execution_count})")
        print(f"   Query:  {job.query}")
        print(f"   Result: {execution.result[:120]}")
        print(f"   Cost:   ${execution.cost_usd:.4f} | Duration: {execution.duration_ms:.0f}ms")

    scheduler.on_execution.append(on_exec)

    # ── Schedule jobs ──
    print("\n📋 Scheduling jobs...")

    job1 = scheduler.schedule(
        weather_agent,
        "What's the weather in Tokyo right now?",
        interval="30s",
        max_executions=3,
    )
    print(f"   ✓ Job 1: '{job1.agent_name}' every 30s (max 3 runs) — ID: {job1.job_id}")

    job2 = scheduler.schedule(
        report_agent,
        "Calculate: 42 * 17 + 99",
        interval="1m",
        max_executions=2,
    )
    print(f"   ✓ Job 2: '{job2.agent_name}' every 60s (max 2 runs) — ID: {job2.job_id}")

    # ── Start scheduler ──
    print(f"\n🚀 Starting scheduler... (will run for ~2 minutes)")
    print(f"   Active jobs: {scheduler.active_jobs}")
    print(f"   Max concurrent: {scheduler.max_concurrent}")
    print("-" * 60)

    scheduler.start()

    # Wait and show status periodically
    try:
        for tick in range(24):  # 24 * 5s = 2 minutes
            time.sleep(5)
            overview = scheduler.get_overview()
            ts = time.strftime("%H:%M:%S")
            total_execs = overview["total_executions"]
            total_cost = overview["total_cost"]

            # Check if all jobs are done
            all_done = all(j.is_finished() for j in scheduler.list_jobs())
            if all_done:
                print(f"\n✅ [{ts}] All jobs completed!")
                break

            # Periodic status
            if tick % 2 == 1:
                print(f"\n📊 [{ts}] Status: {overview['active_jobs']} active, "
                      f"{total_execs} total runs, ${total_cost:.4f} cost")
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")

    # ── Stop and report ──
    scheduler.stop()

    print("\n" + "=" * 60)
    print("📊 Scheduler Summary")
    print("=" * 60)
    for job in scheduler.list_jobs():
        print(f"\n   Job: {job.agent_name} ({job.job_id})")
        print(f"   Status: {job.status.value}")
        print(f"   Executions: {job.execution_count}/{job.max_executions or '∞'}")
        total_cost = sum(e.cost_usd for e in job.history)
        print(f"   Total cost: ${total_cost:.4f}")
        if job.history:
            print(f"   Last result: {job.history[-1].result[:100]}")
    print("=" * 60)

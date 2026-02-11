"""AgentOS Scheduler — run agents on schedules.

Usage:
    from agentos.scheduler import AgentScheduler
    from agentos.core.agent import Agent

    scheduler = AgentScheduler()
    agent = Agent(name="bot", tools=[...])

    scheduler.schedule(agent, "Check weather", interval="5m")
    scheduler.schedule(agent, "Daily report", cron="0 9 * * *")

    scheduler.start()
"""

from agentos.scheduler.scheduler import AgentScheduler
from agentos.scheduler.job import Job, JobStatus, JobExecution, parse_interval

__all__ = [
    "AgentScheduler",
    "Job",
    "JobStatus",
    "JobExecution",
    "parse_interval",
]

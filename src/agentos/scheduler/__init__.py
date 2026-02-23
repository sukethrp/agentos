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

_default_scheduler: AgentScheduler | None = None


def get_scheduler() -> AgentScheduler:
    global _default_scheduler
    if _default_scheduler is None:
        _default_scheduler = AgentScheduler(max_concurrent=3)
        _default_scheduler.start()
    return _default_scheduler


__all__ = [
    "AgentScheduler",
    "Job",
    "JobStatus",
    "JobExecution",
    "parse_interval",
    "get_scheduler",
]

"""AgentScheduler — run agents on schedules using threading."""

from __future__ import annotations
import io
import sys
import threading
import time
from agentos.core.agent import Agent
from agentos.core.tool import Tool
from agentos.scheduler.job import Job, JobStatus, parse_interval


class AgentScheduler:
    """Schedule agents to run at intervals or cron expressions.

    Usage:
        scheduler = AgentScheduler()
        agent = Agent(name="weather-bot", tools=[weather_tool()])

        scheduler.schedule(agent, "What's the weather in Tokyo?", interval="5m")
        scheduler.schedule(agent, "Daily report", cron="0 9 * * *")

        scheduler.start()
        # ... later ...
        scheduler.stop()
    """

    def __init__(self, max_concurrent: int = 3, tick_interval: float = 1.0):
        self.max_concurrent = max_concurrent
        self.tick_interval = tick_interval

        self._jobs: dict[str, Job] = {}
        self._agents: dict[str, Agent] = {}  # job_id -> agent
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._active_count = 0

        # Callbacks
        self.on_execution: list = []  # called with (job, execution)

    # ── Schedule ──

    def schedule(
        self,
        agent: Agent,
        query: str,
        interval: str = "",
        cron: str = "",
        max_executions: int = 0,
        job_id: str = "",
    ) -> Job:
        """Schedule an agent to run periodically.

        Args:
            agent: The Agent instance to run.
            query: The query/prompt to send to the agent.
            interval: Interval string like "5m", "1h", "30s", "1d".
            cron: Cron expression like "0 9 * * *" (minute hour dom month dow).
            max_executions: Stop after N executions (0 = unlimited).
            job_id: Custom job ID (auto-generated if empty).

        Returns:
            The created Job object.
        """
        interval_secs = parse_interval(interval) if interval else 0.0

        if not interval_secs and not cron:
            raise ValueError("Must provide either interval or cron expression")

        job = Job(
            agent_name=agent.config.name,
            query=query,
            interval_seconds=interval_secs,
            cron_expression=cron,
            max_executions=max_executions,
        )
        if job_id:
            job.job_id = job_id

        with self._lock:
            self._jobs[job.job_id] = job
            self._agents[job.job_id] = agent

        return job

    def schedule_from_config(
        self,
        agent_name: str,
        model: str,
        query: str,
        tools: list[Tool] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        interval: str = "",
        cron: str = "",
        max_executions: int = 0,
    ) -> Job:
        """Schedule by creating an agent from config (used by web API)."""
        agent = Agent(
            name=agent_name,
            model=model,
            tools=tools or [],
            system_prompt=system_prompt,
        )
        return self.schedule(agent, query, interval=interval, cron=cron, max_executions=max_executions)

    # ── Control ──

    def start(self) -> None:
        """Start the scheduler loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the scheduler and wait for running jobs."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Job management ──

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def pause_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status not in (JobStatus.CANCELLED,):
            job.status = JobStatus.PAUSED
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PAUSED:
            job.status = JobStatus.PENDING
            job.update_next_run()
            return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.CANCELLED
            with self._lock:
                self._agents.pop(job_id, None)
            return True
        return False

    def delete_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED
            with self._lock:
                self._jobs.pop(job_id, None)
                self._agents.pop(job_id, None)
            return True
        return False

    # ── Info ──

    @property
    def job_count(self) -> int:
        return len(self._jobs)

    @property
    def active_jobs(self) -> int:
        return sum(
            1 for j in self._jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.RUNNING)
        )

    def get_overview(self) -> dict:
        return {
            "running": self._running,
            "total_jobs": self.job_count,
            "active_jobs": self.active_jobs,
            "max_concurrent": self.max_concurrent,
            "total_executions": sum(j.execution_count for j in self._jobs.values()),
            "total_cost": round(
                sum(e.cost_usd for j in self._jobs.values() for e in j.history), 6
            ),
        }

    # ── Internal loop ──

    def _run_loop(self) -> None:
        """Main scheduler loop. Checks for due jobs and runs them."""
        while self._running:
            due_jobs = []
            with self._lock:
                for job_id, job in self._jobs.items():
                    if job.is_due() and not job.is_finished():
                        if self._active_count < self.max_concurrent:
                            due_jobs.append(job_id)

            for job_id in due_jobs:
                self._execute_job(job_id)

            # Clean up finished jobs (mark them)
            for job in self._jobs.values():
                if job.is_finished() and job.status != JobStatus.CANCELLED:
                    job.status = JobStatus.COMPLETED

            time.sleep(self.tick_interval)

    def _execute_job(self, job_id: str) -> None:
        """Execute a job in a separate thread."""
        job = self._jobs.get(job_id)
        agent = self._agents.get(job_id)
        if not job or not agent:
            return

        job.status = JobStatus.RUNNING
        job.last_run = time.time()

        with self._lock:
            self._active_count += 1

        def run():
            try:
                # Capture stdout to avoid cluttering scheduler output
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()

                msg = agent.run(job.query)

                sys.stdout = old_stdout

                result = msg.content or ""
                cost = sum(e.cost_usd for e in agent.events)
                tokens = sum(e.tokens_used for e in agent.events)

                job.record_execution(result, cost=cost, tokens=tokens)
                job.status = JobStatus.PENDING
                job.update_next_run()

                for cb in self.on_execution:
                    try:
                        cb(job, job.history[-1])
                    except Exception:
                        pass

            except Exception as e:
                sys.stdout = old_stdout if 'old_stdout' in dir() else sys.__stdout__
                job.record_execution("", error=str(e))
                job.status = JobStatus.PENDING
                job.update_next_run()
            finally:
                with self._lock:
                    self._active_count -= 1

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

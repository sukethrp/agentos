from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from agentos.tools import get_builtin_tools
from agentos.web.deps import get_scheduler

router = APIRouter(tags=["scheduler"])


class ScheduleRequest(BaseModel):
    agent_name: str = "scheduled-agent"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    query: str
    tools: list[str] = []
    interval: str = ""
    cron: str = ""
    max_executions: int = 0


@router.get("/api/scheduler/jobs")
def list_scheduler_jobs():
    """List all scheduled jobs."""
    scheduler = get_scheduler()
    return {
        "overview": scheduler.get_overview(),
        "jobs": [j.to_dict() for j in scheduler.list_jobs()],
    }


@router.post("/api/scheduler/create")
def create_scheduler_job(req: ScheduleRequest):
    """Create a new scheduled job."""
    available_tools = get_builtin_tools()
    agent_tools = [available_tools[t] for t in req.tools if t in available_tools]
    scheduler = get_scheduler()

    try:
        job = scheduler.schedule_from_config(
            agent_name=req.agent_name,
            model=req.model,
            query=req.query,
            tools=agent_tools,
            system_prompt=req.system_prompt,
            interval=req.interval,
            cron=req.cron,
            max_executions=req.max_executions,
        )
        return {"status": "created", "job": job.to_dict()}
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@router.delete("/api/scheduler/delete/{job_id}")
def delete_scheduler_job(job_id: str):
    """Delete a scheduled job."""
    if get_scheduler().delete_job(job_id):
        return {"status": "deleted", "job_id": job_id}
    return {"status": "error", "message": f"Job {job_id} not found"}


@router.post("/api/scheduler/pause/{job_id}")
def pause_scheduler_job(job_id: str):
    """Pause a scheduled job."""
    if get_scheduler().pause_job(job_id):
        return {"status": "paused", "job_id": job_id}
    return {"status": "error", "message": f"Cannot pause job {job_id}"}


@router.post("/api/scheduler/resume/{job_id}")
def resume_scheduler_job(job_id: str):
    """Resume a paused job."""
    if get_scheduler().resume_job(job_id):
        return {"status": "resumed", "job_id": job_id}
    return {"status": "error", "message": f"Cannot resume job {job_id}"}

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agentos.scheduler import get_scheduler
from agentos.core.agent import Agent

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


class AddJobRequest(BaseModel):
    agent_name: str = "scheduled-agent"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    query: str
    interval: str = ""
    cron: str = ""
    max_executions: int = 0
    job_id: str = ""


@router.post("/jobs")
async def scheduler_add_job(req: AddJobRequest) -> dict:
    scheduler = get_scheduler()
    agent = Agent(
        name=req.agent_name,
        model=req.model,
        tools=[],
        system_prompt=req.system_prompt,
    )
    job = scheduler.schedule(
        agent,
        req.query,
        interval=req.interval,
        cron=req.cron,
        max_executions=req.max_executions,
        job_id=req.job_id or "",
    )
    return {"job_id": job.job_id}


@router.get("/jobs")
async def scheduler_list_jobs() -> list:
    jobs = get_scheduler().list_jobs()
    return [
        {
            "job_id": j.job_id,
            "agent_name": j.agent_name,
            "query": j.query,
            "status": j.status.value,
            "execution_count": j.execution_count,
        }
        for j in jobs
    ]


@router.delete("/jobs/{job_id}")
async def scheduler_remove_job(job_id: str) -> dict:
    ok = get_scheduler().delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "removed", "job_id": job_id}


@router.post("/jobs/{job_id}/pause")
async def scheduler_pause_job(job_id: str) -> dict:
    ok = get_scheduler().pause_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "paused", "job_id": job_id}


@router.post("/jobs/{job_id}/resume")
async def scheduler_resume_job(job_id: str) -> dict:
    ok = get_scheduler().resume_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "resumed", "job_id": job_id}

from __future__ import annotations
import threading as _sim_threading
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agentos.simulation import (
    SimulatedWorld,
    WorldConfig,
    TrafficPattern,
    SimulationReport,
    ALL_PERSONAS,
)
from agentos.tools import get_builtin_tools

router = APIRouter(tags=["simulation"])

_sim_world: SimulatedWorld | None = None
_sim_report: SimulationReport | None = None
_sim_thread: _sim_threading.Thread | None = None

class _SimRunBody(BaseModel):
    total: int = 50
    concurrency: int = 5
    pattern: str = "burst"
    system_prompt: str = "You are a helpful customer support assistant."
    pass_threshold: float = 6.0
@router.post("/api/simulation/run")
def simulation_run(body: _SimRunBody) -> dict:
    global _sim_world, _sim_report, _sim_thread
    if _sim_world and _sim_world.running:
        return {"status": "already_running"}

    from agentos.core.agent import Agent

    agent = Agent(
        name="sim-agent",
        model="gpt-4o-mini",
        system_prompt=body.system_prompt,
        tools=list(get_builtin_tools().values()),
    )
    cfg = WorldConfig(
        total_interactions=min(body.total, 200),
        concurrency=min(body.concurrency, 20),
        traffic_pattern=TrafficPattern(body.pattern),
        requests_per_second=3.0,
        use_llm_judge=False,
        pass_threshold=body.pass_threshold,
        quiet=True,
    )
    _sim_world = SimulatedWorld(agent, cfg)
    _sim_report = None

    def _run():
        global _sim_report
        _sim_report = _sim_world.run()  # type: ignore[union-attr]

    _sim_thread = _sim_threading.Thread(target=_run, daemon=True)
    _sim_thread.start()
    return {"status": "started", "total": cfg.total_interactions}


@router.get("/api/simulation/status")
def simulation_status() -> dict:
    if not _sim_world:
        return {"running": False, "progress": 0, "completed": 0, "total": 0}
    return {
        "running": _sim_world.running,
        "progress": _sim_world.progress,
        "completed": _sim_world._progress,
        "total": _sim_world.config.total_interactions,
    }


@router.get("/api/simulation/report")
def simulation_report() -> dict:
    if _sim_report:
        return _sim_report.to_dict()
    return {"total_interactions": 0}


@router.get("/api/simulation/personas")
def simulation_personas() -> list:
    return [p.to_dict() for p in ALL_PERSONAS]


@router.get("/sandbox/runs/{run_id}/report")
def sandbox_run_report(run_id: str):
    from agentos.sandbox.simulation_runner import get_run_report

    report = get_run_report(run_id)
    if report is None:
        return JSONResponse({"error": "run not found"}, status_code=404)
    return report


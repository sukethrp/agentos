from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from agentos.sandbox.simulation_runner import SimulationRunner, get_run_report
from agentos.sandbox.comparison import ComparisonReport
from agentos.sandbox.evaluation_scenario import EvaluationScenario
from agentos.core.agent import Agent

router = APIRouter(prefix="/sandbox", tags=["sandbox"])

_runner = SimulationRunner()
_scenarios: list[EvaluationScenario] = []


class RunBatchRequest(BaseModel):
    agent_name: str = "sandbox-agent"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    concurrency: int = 10
    scenario_ids: list[str] = Field(default_factory=list)


class CreateScenarioRequest(BaseModel):
    scenario_id: str
    input: str
    expected_output: str
    rubric: str
    tags: list[str] = Field(default_factory=list)


@router.post("/run")
async def sandbox_run(req: RunBatchRequest) -> dict:
    scenarios = [
        s
        for s in _scenarios
        if not req.scenario_ids or s.scenario_id in req.scenario_ids
    ]
    if not scenarios:
        raise HTTPException(status_code=400, detail="No scenarios match")
    agent = Agent(
        name=req.agent_name,
        model=req.model,
        tools=[],
        system_prompt=req.system_prompt,
    )
    run_id = await _runner.run_batch(scenarios, agent, concurrency=req.concurrency)
    return {"run_id": run_id}


@router.get("/runs/{run_id}/report")
async def sandbox_run_report(run_id: str, run_id_b: str | None = None) -> dict:
    if run_id_b:
        return ComparisonReport.generate(run_id, run_id_b)
    report = get_run_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return report


@router.post("/scenarios")
async def sandbox_create_scenario(req: CreateScenarioRequest) -> dict:
    scenario = EvaluationScenario(
        scenario_id=req.scenario_id,
        input=req.input,
        expected_output=req.expected_output,
        rubric=req.rubric,
        tags=req.tags,
    )
    _scenarios.append(scenario)
    return {"scenario_id": scenario.scenario_id}


@router.get("/scenarios")
async def sandbox_list_scenarios() -> list:
    return [s.model_dump() for s in _scenarios]

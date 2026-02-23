from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from agentos.teams.runner import TeamRunner
from agentos.teams.planner import PlannerAgent
from agentos.teams.dag import WorkflowDAG
from agentos.core.agent import Agent

router = APIRouter(prefix="/teams", tags=["teams"])


class RunTeamRequest(BaseModel):
    team_id: str = "team-1"
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    agents: dict[str, dict] = Field(default_factory=dict)
    initial_input: str = ""


class PlanRequest(BaseModel):
    goal: str
    registered_agents: list[str] = Field(default_factory=list)


@router.post("/run")
async def teams_run(req: RunTeamRequest) -> dict:
    if not req.nodes or not req.edges:
        raise HTTPException(status_code=400, detail="nodes and edges required")
    nodes = [n if "id" in n else {**n, "id": f"n{i}"} for i, n in enumerate(req.nodes)]
    dag = WorkflowDAG(nodes=nodes, edges=req.edges)
    agents_map = {}
    for aid, acfg in req.agents.items():
        agents_map[aid] = Agent(
            name=acfg.get("name", aid),
            model=acfg.get("model", "gpt-4o-mini"),
            tools=[],
            system_prompt=acfg.get("system_prompt", "You are a helpful assistant."),
        )
    runner = TeamRunner(team_id=req.team_id, agents=agents_map)
    outputs = await runner.execute(dag, req.initial_input)
    return {"outputs": outputs}


@router.post("/plan")
async def teams_plan(req: PlanRequest) -> list:
    planner = PlannerAgent(registered_agents=req.registered_agents or None)
    return planner.plan(req.goal)

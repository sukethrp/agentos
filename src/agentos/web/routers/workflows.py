from __future__ import annotations
import uuid as _uuid
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agentos.web.deps import get_workflows_store

router = APIRouter(tags=["workflows"])

class WorkflowCreate(BaseModel):
    name: str
    dag: str

@router.post("/workflows/")
def workflows_create(req: WorkflowCreate):
    wid = str(_uuid.uuid4())
    get_workflows_store()[wid] = {"id": wid, "name": req.name, "dag": req.dag}
    return {"id": wid, "name": req.name}


@router.post("/workflows/{workflow_id}/run")
def workflows_run(workflow_id: str):
    if workflow_id not in get_workflows_store():
        return JSONResponse({"error": "workflow not found"}, status_code=404)
    import asyncio
    from agentos.teams.dag import WorkflowDAG
    from agentos.teams.runner import TeamRunner
    from agentos.core.agent import Agent
    import yaml

    w = get_workflows_store()[workflow_id]
    try:
        data = yaml.safe_load(w["dag"])
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        dag = WorkflowDAG(nodes=nodes, edges=edges)
        agents = {
            n.get("agent_id", "agent"): Agent(name=n.get("agent_id", "agent"))
            for n in nodes
        }
        runner = TeamRunner(workflow_id, agents)
        asyncio.run(runner.execute(dag, ""))
    except Exception:
        pass
    return {"status": "started", "workflow_id": workflow_id}


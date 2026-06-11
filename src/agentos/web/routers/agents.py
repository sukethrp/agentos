from __future__ import annotations
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from agentos.auth import User, get_optional_user
from agentos.auth.usage import usage_tracker
from agentos.core.ab_testing import ABTest
from agentos.monitor.store import store
from agentos.tools import get_builtin_tools

router = APIRouter(tags=["agents"])

class RunRequest(BaseModel):
    name: str = "web-agent"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    query: str = ""
    tools: list[str] = []
    temperature: float = 0.7
    budget_limit: float = 5.0


class ABTestAgentConfig(BaseModel):
    name: str = "agent"
    model: str = "gpt-4o-mini"
    system_prompt: str
    temperature: float = 0.7
    tools: list[str] = []


class ABTestRequest(BaseModel):
    agent_a: ABTestAgentConfig
    agent_b: ABTestAgentConfig
    queries: list[str]
    num_runs: int = 5

@router.post("/api/run")
def run_agent(req: RunRequest, current_user: User | None = Depends(get_optional_user)):
    """Run an agent from the web UI.  Auth is optional — anonymous use is allowed."""
    from agentos.core.agent import Agent

    available_tools = get_builtin_tools()
    agent_tools = [available_tools[t] for t in req.tools if t in available_tools]

    agent = Agent(
        name=req.name,
        model=req.model,
        tools=agent_tools,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
    )

    import io
    import sys

    old = sys.stdout
    sys.stdout = io.StringIO()
    msg = agent.run(req.query)
    terminal_output = sys.stdout.getvalue()
    sys.stdout = old

    for e in agent.events:
        store.log_event(e)

    cost = sum(e.cost_usd for e in agent.events)
    tokens = sum(e.tokens_used for e in agent.events)
    tools_used = [
        e.data.get("tool", "") for e in agent.events if e.event_type == "tool_call"
    ]

    if current_user:
        usage_tracker.log_usage(current_user.id, tokens=tokens, cost=cost)

    return {
        "response": msg.content,
        "cost": round(cost, 6),
        "tokens": tokens,
        "tools_used": tools_used,
        "terminal": terminal_output,
    }


@router.post("/api/ab-test")
def run_ab_test(
    req: ABTestRequest, current_user: User | None = Depends(get_optional_user)
):
    """Run an A/B test between two agent configs using the Sandbox judge."""
    from agentos.core.agent import Agent

    queries = [q.strip() for q in req.queries if q.strip()]
    if not queries:
        return {
            "status": "error",
            "message": "At least one non-empty query is required",
        }

    available_tools = get_builtin_tools()

    def build_agent(cfg: ABTestAgentConfig) -> Agent:
        agent_tools = [available_tools[t] for t in cfg.tools if t in available_tools]
        return Agent(
            name=cfg.name,
            model=cfg.model,
            tools=agent_tools,
            system_prompt=cfg.system_prompt,
            temperature=cfg.temperature,
        )

    agent_a = build_agent(req.agent_a)
    agent_b = build_agent(req.agent_b)

    tester = ABTest(agent_a, agent_b)
    report = tester.run_test(queries, num_runs=req.num_runs)

    return {"status": "ok", "report": report.model_dump()}


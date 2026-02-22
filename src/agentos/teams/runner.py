from __future__ import annotations
import asyncio
from typing import Any
from agentos.core.agent import Agent
from agentos.teams.dag import WorkflowDAG
from agentos.governance.guardrails import GovernanceEngine
from agentos.governance.budget import BudgetGuard
from agentos.monitor.ws_manager import broadcast_team_node_event


def _safe_eval(expr: str, outputs: dict[str, str]) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}}, {"outputs": outputs}))
    except Exception:
        return False


class TeamRunner:
    def __init__(
        self,
        team_id: str,
        agents: dict[str, Agent],
        governance: GovernanceEngine | None = None,
    ):
        self.team_id = team_id
        self._agents = agents
        self._gov = governance or GovernanceEngine(
            agent_name=team_id,
            budget=BudgetGuard(max_per_day=50.0, max_total=50.0),
        )

    async def execute(self, dag: WorkflowDAG, initial_input: str = "") -> dict[str, str]:
        outputs: dict[str, str] = {}
        order = dag.topological_order()
        ready: dict[str, set[str]] = {n: set(dag.predecessors(n)) for n in order}

        async def run_node(node_id: str, task_input: str) -> str:
            await broadcast_team_node_event(self.team_id, node_id, "running", "")
            agent = self._agents.get(dag.node_data(node_id).get("agent_id", ""))
            if not agent:
                out = ""
                await broadcast_team_node_event(self.team_id, node_id, "error", "agent not found")
                return out
            est_cost = 0.01
            check = self._gov.check_tool_call(f"llm:{node_id}", estimated_cost=est_cost)
            if not check.allowed:
                out = f"BLOCKED: {check.message}"
                await broadcast_team_node_event(self.team_id, node_id, "blocked", out)
                return out
            loop = asyncio.get_event_loop()
            msg = await loop.run_in_executor(None, lambda: agent.run(task_input))
            out = msg.content or ""
            cost = sum(e.cost_usd for e in agent.events)
            self._gov.record_action(f"llm:{node_id}", cost)
            await broadcast_team_node_event(self.team_id, node_id, "done", out)
            return out

        completed = set()
        while len(completed) < len(order):
            batch = []
            for n in order:
                if n in completed:
                    continue
                preds = dag.predecessors(n)
                required = set()
                for p in preds:
                    edge_data = dag.edge_data(p, n)
                    cond = edge_data.get("condition_expr")
                    if not cond or _safe_eval(cond, outputs):
                        required.add(p)
                if required.issubset(completed):
                    batch.append(n)
            if not batch:
                break
            tasks_input = []
            for n in batch:
                preds = dag.predecessors(n)
                if not preds:
                    tasks_input.append(initial_input)
                else:
                    parts = [outputs.get(p, "") for p in preds]
                    tasks_input.append("\n\n".join(parts))
            results = await asyncio.gather(*[run_node(n, ti) for n, ti in zip(batch, tasks_input)])
            for n, r in zip(batch, results):
                outputs[n] = r
                completed.add(n)
        return outputs


class ResultAggregator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model

    def merge(self, outputs: dict[str, str], goal: str = "") -> dict[str, Any]:
        from agentos.providers.router import call_model
        parts = [f"[{k}]: {v}" for k, v in outputs.items()]
        prompt = f"Synthesize these subtask outputs into a final structured response.\n\nGoal: {goal}\n\nOutputs:\n" + "\n\n".join(parts) + "\n\nReturn valid JSON with keys: summary, details, recommendations (or similar)."
        msg, _ = call_model(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            temperature=0.3,
        )
        text = (msg.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            import json
            return json.loads(text)
        except Exception:
            return {"summary": text, "details": outputs, "recommendations": []}

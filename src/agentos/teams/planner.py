from __future__ import annotations
import json
from agentos.providers.router import call_model


PLANNER_PROMPT = """Given this goal, produce a JSON list of subtasks. Each subtask has: task (str), agent_id (str), depends_on (list of indices 0-based, or empty).

Goal: {goal}

Available agents: {agents}

Return ONLY valid JSON array, e.g. [{{"task": "...", "agent_id": "...", "depends_on": []}}, ...]
"""


class PlannerAgent:
    def __init__(
        self, model: str = "gpt-4o-mini", registered_agents: list[str] | None = None
    ):
        self._model = model
        self._registered = set(registered_agents or [])

    def register_agents(self, agent_ids: list[str]) -> None:
        self._registered.update(agent_ids)

    def plan(self, goal: str) -> list[dict]:
        agents_str = ", ".join(sorted(self._registered)) if self._registered else "none"
        prompt = PLANNER_PROMPT.format(goal=goal, agents=agents_str)
        msg, _ = call_model(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            temperature=0.0,
        )
        text = (msg.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            raw = json.loads(text)
            if not isinstance(raw, list):
                return []
            validated = []
            for i, item in enumerate(raw):
                if not isinstance(item, dict):
                    continue
                task = item.get("task", "")
                agent_id = str(item.get("agent_id", ""))
                depends_on = item.get("depends_on", [])
                if isinstance(depends_on, list):
                    depends_on = [
                        int(x) for x in depends_on if isinstance(x, (int, float))
                    ]
                else:
                    depends_on = []
                if not self._registered or agent_id in self._registered:
                    validated.append(
                        {"task": task, "agent_id": agent_id, "depends_on": depends_on}
                    )
            return validated
        except (json.JSONDecodeError, ValueError):
            return []

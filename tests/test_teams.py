from __future__ import annotations
from unittest.mock import patch
import pytest
from agentos.core.types import Message, Role
from agentos.teams.planner import PlannerAgent
from agentos.teams.runner import TeamRunner
from agentos.teams.dag import WorkflowDAG


def test_planner_decomposes_goal():
    import json

    mock_subtasks = [
        {"task": "Research topic", "agent_id": "researcher", "depends_on": []},
        {"task": "Write summary", "agent_id": "writer", "depends_on": [0]},
    ]
    mock_content = json.dumps(mock_subtasks)

    with patch("agentos.teams.planner.call_model") as mock_call:
        mock_call.return_value = (
            Message(role=Role.ASSISTANT, content=mock_content),
            None,
        )
        planner = PlannerAgent(registered_agents=["researcher", "writer"])
        result = planner.plan("Write a report on AI")
        assert len(result) == 2
        assert result[0]["task"] == "Research topic"
        assert result[0]["agent_id"] == "researcher"
        assert result[1]["task"] == "Write summary"
        assert result[1]["agent_id"] == "writer"


@pytest.mark.asyncio
async def test_team_runner_sequential():
    outputs = []

    class MockAgent:
        def __init__(self, name: str):
            self.config = type("C", (), {"name": name})()
            self.tools = []
            self.events = []

        def run(self, inp: str):
            out = f"out_{self.config.name}"
            outputs.append((self.config.name, inp))
            return Message(role=Role.ASSISTANT, content=out)

    dag = WorkflowDAG(
        nodes=[
            {"id": "n1", "agent_id": "a1"},
            {"id": "n2", "agent_id": "a2"},
        ],
        edges=[{"source": "n1", "target": "n2"}],
    )
    agents = {"a1": MockAgent("a1"), "a2": MockAgent("a2")}
    runner = TeamRunner(team_id="t1", agents=agents)
    with patch("agentos.teams.runner.broadcast_team_node_event"):
        result = await runner.execute(dag, "start")
    assert result["n1"] == "out_a1"
    assert result["n2"] == "out_a2"
    assert outputs[0][0] == "a1"
    assert outputs[1][0] == "a2"
    assert outputs[0][1] == "start"
    assert outputs[1][1] == "out_a1"


@pytest.mark.asyncio
async def test_team_runner_parallel():
    order = []

    class MockAgent:
        def __init__(self, name: str):
            self.config = type("C", (), {"name": name})()
            self.tools = []
            self.events = []

        def run(self, inp: str):
            order.append(self.config.name)
            return Message(role=Role.ASSISTANT, content=f"out_{self.config.name}")

    dag = WorkflowDAG(
        nodes=[
            {"id": "n1", "agent_id": "a1"},
            {"id": "n2", "agent_id": "a2"},
            {"id": "n3", "agent_id": "a3"},
        ],
        edges=[
            {"source": "n1", "target": "n3"},
            {"source": "n2", "target": "n3"},
        ],
    )

    class A3:
        def __init__(self):
            self.config = type("C", (), {"name": "a3"})()
            self.tools = []
            self.events = []

        def run(self, inp: str):
            order.append("a3")
            return Message(role=Role.ASSISTANT, content="out_a3")

    agents = {"a1": MockAgent("a1"), "a2": MockAgent("a2"), "a3": A3()}
    runner = TeamRunner(team_id="t1", agents=agents)
    with patch("agentos.teams.runner.broadcast_team_node_event"):
        result = await runner.execute(dag, "start")
    assert "n1" in result
    assert "n2" in result
    assert "n3" in result
    assert "a1" in order
    assert "a2" in order
    assert "a3" in order

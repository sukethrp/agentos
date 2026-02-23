from __future__ import annotations
import asyncio
import pytest
from agentos.mesh.mesh_router import MeshRouter, MeshMessage


@pytest.fixture
def mesh_router():
    return MeshRouter()


@pytest.mark.asyncio
async def test_register_agent(mesh_router):
    await mesh_router.send_message("agent_a", "agent_b", {"ping": 1})
    await mesh_router.send_message("agent_b", "agent_a", {"pong": 1})
    agents = mesh_router.registered_agents()
    assert "agent_a" in agents
    assert "agent_b" in agents
    assert len(agents) == 2


@pytest.mark.asyncio
async def test_send_message(mesh_router):
    await mesh_router.send_message("agent_a", "agent_b", {"data": "hello"})
    q = mesh_router._get_queue("agent_b")
    msg = await asyncio.wait_for(q.get(), timeout=1.0)
    assert msg.sender == "agent_a"
    assert msg.receiver == "agent_b"
    assert msg.payload == {"data": "hello"}


@pytest.mark.asyncio
async def test_broadcast(mesh_router):
    await mesh_router.send_message("agent_a", "agent_b", {})
    await mesh_router.send_message("agent_a", "agent_c", {})
    msgs = await mesh_router.broadcast("agent_a", "alerts", {"alert": "test"})
    assert len(msgs) == 2
    for m in msgs:
        assert isinstance(m, MeshMessage)
        assert m.topic == "alerts"
        assert m.payload == {"alert": "test"}


@pytest.mark.asyncio
async def test_subscribe_and_receive(mesh_router):
    received = []

    async def handler(msg: MeshMessage):
        received.append(msg)

    mesh_router.subscribe("agent_b", "alerts", handler)
    await mesh_router.send_message(
        "agent_a", "agent_b", {"alert": "fire"}, topic="alerts"
    )
    await asyncio.sleep(0.1)
    assert len(received) == 1
    assert received[0].payload == {"alert": "fire"}
    assert received[0].topic == "alerts"

"""AgentOS Mesh — agent-to-agent communication protocol.

Usage::

    from agentos.mesh import AgentMesh, SharedContext

    mesh = AgentMesh(name="my-team")
    mesh.add(agent_a)
    mesh.add(agent_b)

    # Orchestrator pattern
    result = mesh.run_orchestrated(coordinator=agent_a, task="...")

    # Peer-to-peer
    mesh.enable_peer_delegation()
    result = agent_a.run("Delegate the writing part to agent_b")
"""

from agentos.mesh.mesh import AgentMesh, MeshCostTracker
from agentos.mesh.protocol import (
    AgentMessage,
    DelegationRequest,
    DelegationResult,
    SharedContext,
)
from agentos.mesh.registry import AgentRegistry, default_registry

__all__ = [
    "AgentMesh",
    "AgentMessage",
    "AgentRegistry",
    "DelegationRequest",
    "DelegationResult",
    "MeshCostTracker",
    "SharedContext",
    "default_registry",
]

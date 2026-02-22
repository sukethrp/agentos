from __future__ import annotations
from fastapi import APIRouter
from agentos.mesh.mesh_router import get_mesh_router

router = APIRouter(prefix="/mesh", tags=["mesh"])


@router.get("/health")
async def mesh_health() -> dict:
    mr = get_mesh_router()
    return {
        "registered_agents": mr.registered_agents(),
        "queue_depths": mr.queue_depths(),
    }

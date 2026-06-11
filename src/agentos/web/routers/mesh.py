from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agentos.mesh.protocol import MeshMessage as _MeshMsg, MeshIdentity as _MeshId
from agentos.mesh.discovery import get_registry as _get_mesh_registry
from agentos.mesh.transaction import get_ledger as _get_mesh_ledger
from agentos.mesh.server import (
    handle_message as _mesh_handle,
    get_node as _get_mesh_node,
    init_node as _mesh_init_node,
)

router = APIRouter(tags=["mesh"])

try:
    _mesh_init_node(
        mesh_id="platform@agentos.local",
        display_name="AgentOS Platform",
        organisation="AgentOS",
        capabilities=["ping", "negotiate", "transact", "verify"],
        endpoint_url="http://localhost:8000/api/mesh",
    )
except Exception:
    pass

class _MeshRegisterBody(BaseModel):
    mesh_id: str
    display_name: str = ""
    public_key: str = ""
    capabilities: list[str] = []
    endpoint_url: str = ""
    organisation: str = ""


@router.post("/api/mesh/message")
def mesh_receive_message(msg: _MeshMsg) -> dict:
    resp = _mesh_handle(msg)
    return resp.model_dump()


@router.post("/api/mesh/register")
def mesh_register_agent(body: _MeshRegisterBody) -> dict:
    identity = _MeshId(**body.model_dump())
    _get_mesh_registry().register(identity)
    return {"status": "registered", "mesh_id": identity.mesh_id}


@router.post("/api/mesh/deregister")
def mesh_deregister_agent(body: dict) -> dict:
    mesh_id = body.get("mesh_id", "")
    ok = _get_mesh_registry().deregister(mesh_id)
    if not ok:
        return JSONResponse({"error": f"Agent {mesh_id} not found"}, status_code=404)
    return {"status": "deregistered", "mesh_id": mesh_id}


@router.get("/api/mesh/registry")
def mesh_list_registry() -> list:
    return _get_mesh_registry().to_list()


@router.get("/api/mesh/registry/search")
def mesh_search_registry(
    q: str = "", capability: str = "", organisation: str = ""
) -> list:
    results = _get_mesh_registry().search(
        query=q, capability=capability, organisation=organisation
    )
    return [a.model_dump() for a in results]


@router.get("/api/mesh/transactions")
def mesh_list_transactions() -> list:
    return _get_mesh_ledger().list_transactions()


@router.get("/api/mesh/verify/{tx_id}")
def mesh_verify_transaction(tx_id: str) -> dict:
    return _get_mesh_ledger().verify(tx_id)


@router.get("/api/mesh/stats")
def mesh_stats() -> dict:
    return {
        "registry": _get_mesh_registry().stats(),
        "ledger": _get_mesh_ledger().stats(),
        "node": _get_mesh_node().identity.model_dump(),
    }


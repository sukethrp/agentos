from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from agentos.auth.org_models import Role
from agentos.auth.org_store import (
    create_org,
    get_org,
    list_org_members,
    add_org_member,
    remove_org_member,
)
from agentos.auth.auth import get_current_user
from agentos.auth.models import User

router = APIRouter(tags=["auth"])


class CreateOrgRequest(BaseModel):
    name: str
    monthly_token_cap: int = 0
    monthly_cost_cap_usd: float = 0.0


class AddMemberRequest(BaseModel):
    user_id: str
    role: str = "viewer"


@router.post("/auth/orgs")
def post_orgs(req: CreateOrgRequest, current_user: User = Depends(get_current_user)):
    org = create_org(
        name=req.name,
        monthly_token_cap=req.monthly_token_cap,
        monthly_cost_cap_usd=req.monthly_cost_cap_usd,
    )
    add_org_member(org.org_id, current_user.id, Role.OWNER)
    return {"org_id": org.org_id, "name": org.name}


@router.get("/auth/orgs/{org_id}/members")
def get_org_members(org_id: str, current_user: User = Depends(get_current_user)):
    org = get_org(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Org not found")
    members = list_org_members(org_id)
    return {"members": [m.model_dump() for m in members]}


@router.post("/auth/orgs/{org_id}/members")
def post_org_member(
    org_id: str, req: AddMemberRequest, current_user: User = Depends(get_current_user)
):
    org = get_org(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Org not found")
    try:
        role = Role(req.role.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role")
    mem = add_org_member(org_id, req.user_id, role)
    return {"user_id": mem.user_id, "org_id": mem.org_id, "role": mem.role.value}


@router.delete("/auth/orgs/{org_id}/members/{user_id}")
def delete_org_member(
    org_id: str, user_id: str, current_user: User = Depends(get_current_user)
):
    org = get_org(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Org not found")
    if not remove_org_member(org_id, user_id):
        raise HTTPException(status_code=404, detail="Member not found")
    return {"status": "removed"}

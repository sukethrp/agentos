from __future__ import annotations
from enum import Enum
from pydantic import BaseModel


class Role(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


PERMISSION_MATRIX: dict[tuple[Role, str], bool] = {
    (Role.OWNER, "org:read"): True,
    (Role.OWNER, "org:write"): True,
    (Role.OWNER, "org:delete"): True,
    (Role.OWNER, "members:read"): True,
    (Role.OWNER, "members:write"): True,
    (Role.OWNER, "members:delete"): True,
    (Role.OWNER, "billing:read"): True,
    (Role.OWNER, "billing:write"): True,
    (Role.ADMIN, "org:read"): True,
    (Role.ADMIN, "org:write"): True,
    (Role.ADMIN, "org:delete"): False,
    (Role.ADMIN, "members:read"): True,
    (Role.ADMIN, "members:write"): True,
    (Role.ADMIN, "members:delete"): True,
    (Role.ADMIN, "billing:read"): True,
    (Role.ADMIN, "billing:write"): False,
    (Role.DEVELOPER, "org:read"): True,
    (Role.DEVELOPER, "org:write"): False,
    (Role.DEVELOPER, "org:delete"): False,
    (Role.DEVELOPER, "members:read"): True,
    (Role.DEVELOPER, "members:write"): False,
    (Role.DEVELOPER, "members:delete"): False,
    (Role.DEVELOPER, "billing:read"): True,
    (Role.DEVELOPER, "billing:write"): False,
    (Role.VIEWER, "org:read"): True,
    (Role.VIEWER, "org:write"): False,
    (Role.VIEWER, "org:delete"): False,
    (Role.VIEWER, "members:read"): True,
    (Role.VIEWER, "members:write"): False,
    (Role.VIEWER, "members:delete"): False,
    (Role.VIEWER, "billing:read"): False,
    (Role.VIEWER, "billing:write"): False,
}


def can(role: Role, action: str) -> bool:
    return PERMISSION_MATRIX.get((role, action), False)


class Organization(BaseModel):
    org_id: str
    name: str
    monthly_token_cap: int = 0
    monthly_cost_cap_usd: float = 0.0


class OrgMembership(BaseModel):
    user_id: str
    org_id: str
    role: Role = Role.VIEWER


class ApiKey(BaseModel):
    key_hash: str
    user_id: str
    org_id: str | None = None
    scopes: list[str] = []

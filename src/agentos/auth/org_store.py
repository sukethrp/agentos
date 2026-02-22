from __future__ import annotations
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from agentos.auth.org_models import Organization, OrgMembership, Role, ApiKey


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _default_path() -> str:
    base = Path(__file__).parent
    return str(base / "org_data.json")


_ORG_DATA: dict | None = None
_LOCK = threading.Lock()


def _load() -> dict:
    global _ORG_DATA
    with _LOCK:
        if _ORG_DATA is not None:
            return _ORG_DATA
        path = _default_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    _ORG_DATA = json.load(f)
            except (json.JSONDecodeError, OSError):
                _ORG_DATA = {"orgs": [], "memberships": [], "api_keys": []}
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _ORG_DATA = {"orgs": [], "memberships": [], "api_keys": []}
        return _ORG_DATA


def _save(data: dict) -> None:
    path = _default_path()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_org(name: str, monthly_token_cap: int = 0, monthly_cost_cap_usd: float = 0.0) -> Organization:
    data = _load()
    org_id = str(uuid.uuid4())[:16]
    org = Organization(org_id=org_id, name=name, monthly_token_cap=monthly_token_cap, monthly_cost_cap_usd=monthly_cost_cap_usd)
    data["orgs"].append(org.model_dump())
    _save(data)
    return org


def get_org(org_id: str) -> Organization | None:
    data = _load()
    for o in data.get("orgs", []):
        if o.get("org_id") == org_id:
            return Organization.model_validate(o)
    return None


def list_org_members(org_id: str) -> list[OrgMembership]:
    data = _load()
    return [OrgMembership.model_validate(m) for m in data.get("memberships", []) if m.get("org_id") == org_id]


def add_org_member(org_id: str, user_id: str, role: Role) -> OrgMembership:
    data = _load()
    for m in data.get("memberships", []):
        if m.get("org_id") == org_id and m.get("user_id") == user_id:
            m["role"] = role.value
            _save(data)
            return OrgMembership.model_validate(m)
    mem = OrgMembership(user_id=user_id, org_id=org_id, role=role)
    data.setdefault("memberships", []).append(mem.model_dump())
    _save(data)
    return mem


def remove_org_member(org_id: str, user_id: str) -> bool:
    data = _load()
    ms = data.get("memberships", [])
    orig = len(ms)
    data["memberships"] = [m for m in ms if not (m.get("org_id") == org_id and m.get("user_id") == user_id)]
    if len(data["memberships"]) < orig:
        _save(data)
        return True
    return False


def register_api_key(api_key: str, user_id: str, org_id: str | None = None, scopes: list[str] | None = None) -> ApiKey:
    data = _load()
    kh = _hash_key(api_key)
    for ak in data.get("api_keys", []):
        if ak.get("key_hash") == kh:
            ak["user_id"] = user_id
            ak["org_id"] = org_id
            ak["scopes"] = scopes or []
            _save(data)
            return ApiKey.model_validate(ak)
    ak = ApiKey(key_hash=kh, user_id=user_id, org_id=org_id, scopes=scopes or [])
    data.setdefault("api_keys", []).append(ak.model_dump())
    _save(data)
    return ak


def get_api_key_info(api_key: str) -> ApiKey | None:
    if not api_key:
        return None
    kh = _hash_key(api_key.strip())
    data = _load()
    for ak in data.get("api_keys", []):
        if ak.get("key_hash") == kh:
            return ApiKey.model_validate(ak)
    return None


def check_scope(api_key: str, agent_id: str | None, tool_name: str | None) -> bool:
    info = get_api_key_info(api_key)
    if not info or not info.scopes:
        return True
    scopes = set(info.scopes)
    if "*" in scopes or "agent:*" in scopes or "tool:*" in scopes:
        return True
    if agent_id:
        if f"agent:{agent_id}" in scopes or agent_id in scopes:
            return True
    if tool_name:
        if f"tool:{tool_name}" in scopes or tool_name in scopes:
            return True
    if not agent_id and not tool_name:
        return True
    return False

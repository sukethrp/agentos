"""Authentication helpers for AgentOS.

Provides:
  - create_user(email, name) -> User
  - authenticate(api_key) -> User | None
  - generate_api_key() -> str
  - get_current_user() FastAPI dependency
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from agentos.auth.models import User, default_store


def generate_api_key() -> str:
    """Generate a random, hard-to-guess API key."""
    raw = secrets.token_hex(32)
    # Add a short checksum to make simple typos easier to spot
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"agt_{raw}{digest}"


def create_user(email: str, name: str, is_admin: bool = False, org_id: str | None = None, scopes: list[str] | None = None) -> User:
    api_key = generate_api_key()
    user = default_store.create_user(email=email, name=name, api_key=api_key, is_admin=is_admin)
    try:
        from agentos.auth.org_store import register_api_key
        register_api_key(api_key, user.id, org_id=org_id, scopes=scopes)
    except Exception:
        pass
    return user


def authenticate(api_key: str) -> Optional[User]:
    """Return the user for this API key, or None if not found."""
    if not api_key:
        return None
    return default_store.get_by_api_key(api_key.strip())


def get_current_user(x_api_key: str = Header(..., alias="X-API-Key")) -> User:
    """FastAPI dependency to authenticate the current user via X-API-Key header.

    Raises 401 if the key is missing or invalid.
    """
    user = authenticate(x_api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return user


def get_optional_user(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> Optional[User]:
    """FastAPI dependency that authenticates if a key is provided, but allows anonymous access."""
    if not x_api_key:
        return None
    return authenticate(x_api_key)


def get_user_by_email(email: str) -> Optional[User]:
    """Convenience helper for login flows."""
    return default_store.get_by_email(email)


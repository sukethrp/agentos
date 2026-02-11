"""User models and simple JSON-backed storage for AgentOS auth."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel


class User(BaseModel):
    """Simple user record.

    Note: API keys are stored in plaintext for simplicity. In a real system you
    would hash and salt these.
    """

    id: str
    email: str
    name: str
    api_key: str
    created_at: float
    is_admin: bool = False


@dataclass
class UserStore:
    """JSON file-backed store for users.

    Structure:
        { "users": [ {User}, ... ] }
    """

    path: str

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write({"users": []})

    # ── Low-level IO ──

    def _read(self) -> dict:
        with self._lock:
            if not os.path.exists(self.path):
                return {"users": []}
            with open(self.path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {"users": []}

    def _write(self, data: dict) -> None:
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    # ── CRUD helpers ──

    def list_users(self) -> List[User]:
        data = self._read()
        return [User(**u) for u in data.get("users", [])]

    def get_by_id(self, user_id: str) -> Optional[User]:
        for u in self.list_users():
            if u.id == user_id:
                return u
        return None

    def get_by_email(self, email: str) -> Optional[User]:
        email = email.lower().strip()
        for u in self.list_users():
            if u.email.lower() == email:
                return u
        return None

    def get_by_api_key(self, api_key: str) -> Optional[User]:
        for u in self.list_users():
            if u.api_key == api_key:
                return u
        return None

    def create_user(self, email: str, name: str, api_key: str, is_admin: bool = False) -> User:
        """Create a new user. Raises ValueError if email already exists."""
        if self.get_by_email(email):
            raise ValueError(f"User with email {email} already exists")

        user = User(
            id=uuid.uuid4().hex[:16],
            email=email,
            name=name,
            api_key=api_key,
            created_at=time.time(),
            is_admin=is_admin,
        )
        data = self._read()
        users = data.get("users", [])
        users.append(user.model_dump())
        data["users"] = users
        self._write(data)
        return user

    def update_user(self, user: User) -> User:
        data = self._read()
        users = data.get("users", [])
        updated = False
        for idx, raw in enumerate(users):
            if raw.get("id") == user.id:
                users[idx] = user.model_dump()
                updated = True
                break
        if not updated:
            users.append(user.model_dump())
        data["users"] = users
        self._write(data)
        return user


def default_user_store_path() -> str:
    """Return default path for the user JSON file."""
    # Keep it under the auth package directory
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "users.json")


# Module-level default store
default_store = UserStore(path=default_user_store_path())


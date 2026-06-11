from __future__ import annotations
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from agentos.auth import User, create_user, get_current_user, get_user_by_email
from agentos.auth.usage import usage_tracker

router = APIRouter(tags=["auth"])

class RegisterRequest(BaseModel):
    email: str
    name: str


class LoginRequest(BaseModel):
    email: str


@router.post("/api/auth/register")
def register_user(req: RegisterRequest):
    """Create a new user and return an API key."""
    try:
        user = create_user(email=req.email, name=req.name)
        return {
            "status": "created",
            "user": user,
            "api_key": user.api_key,
        }
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@router.post("/api/auth/login")
def login_user(req: LoginRequest):
    """Login by email — returns existing API key."""
    user = get_user_by_email(req.email)
    if not user:
        return {"status": "error", "message": "User not found"}
    return {
        "status": "ok",
        "user": user,
        "api_key": user.api_key,
    }


@router.get("/api/auth/usage")
def get_my_usage(period: str = "month", current_user: User = Depends(get_current_user)):
    """Return usage stats for the current user."""
    total = usage_tracker.get_usage(current_user.id).to_dict()
    window = usage_tracker.get_usage_by_period(current_user.id, period=period).to_dict()
    return {
        "status": "ok",
        "user_id": current_user.id,
        "period": period,
        "total": total,
        "window": window,
    }


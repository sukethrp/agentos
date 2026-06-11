from __future__ import annotations
from fastapi import APIRouter
from agentos.demo import is_demo_mode
from agentos.monitor.store import store
from agentos.web.deps import get_app

router = APIRouter(prefix="/api", tags=["dashboard"])

@router.get("/config")
def get_config():
    """Return platform configuration, including demo mode status."""
    return {
        "demo_mode": is_demo_mode(),
        "version": get_app().version,
        "provider": "MockProvider (no API keys)" if is_demo_mode() else "auto",
    }


@router.get("/overview")
def overview():
    return store.get_overview()


@router.get("/events")
def get_events(limit: int = 50):
    return store.get_events(limit=limit)


@router.get("/templates")
def get_templates():
    return {
        "templates": [
            {
                "id": "customer-support",
                "name": "Customer Support",
                "description": "Handle inquiries, complaints, tickets",
                "category": "support",
                "icon": "",
            },
            {
                "id": "research-assistant",
                "name": "Research Assistant",
                "description": "Research topics, gather data, analyze",
                "category": "research",
                "icon": "",
            },
            {
                "id": "sales-agent",
                "name": "Sales Agent",
                "description": "Qualify leads, answer product questions",
                "category": "sales",
                "icon": "",
            },
            {
                "id": "code-reviewer",
                "name": "Code Reviewer",
                "description": "Review code for bugs and security",
                "category": "engineering",
                "icon": "",
            },
            {
                "id": "custom",
                "name": "Custom Agent",
                "description": "Build your own from scratch",
                "category": "custom",
                "icon": "",
            },
        ]
    }

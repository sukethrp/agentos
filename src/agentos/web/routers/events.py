from __future__ import annotations
from fastapi import APIRouter
from agentos.events import event_bus
from agentos.web.deps import get_webhook_trigger

router = APIRouter(tags=["events"])

@router.post("/api/webhook/{event_name}")
async def webhook_receiver(event_name: str, body: dict = {}):
    """Receive a webhook POST and emit it through the event bus.

    Example: POST /api/webhook/deploy.completed  {"repo": "myapp", "status": "success"}
    """
    get_webhook_trigger().event_name = f"webhook.{event_name}"
    get_webhook_trigger().fire(data=body, source=f"webhook:{event_name}")
    return {
        "status": "emitted",
        "event": f"webhook.{event_name}",
        "listeners_matched": len(
            [
                lst
                for lst in event_bus.list_listeners()
                if lst.matches(f"webhook.{event_name}")
            ]
        ),
    }


@router.get("/api/events/listeners")
def list_event_listeners():
    """List all registered event listeners."""
    return {
        "overview": event_bus.get_overview(),
        "listeners": [lst.to_dict() for lst in event_bus.list_listeners()],
    }


@router.get("/api/events/history")
def get_event_history(limit: int = 20):
    """Get recent event emission history."""
    return {
        "history": [log.to_dict() for log in event_bus.get_history(limit=limit)],
    }


@router.post("/api/events/emit")
async def emit_event(body: dict = {}):
    """Manually emit an event through the bus.

    Body: {"event_name": "custom.test", "data": {"key": "value"}}
    """
    event_name = body.get("event_name", "custom.manual")
    data = body.get("data", {})
    log = event_bus.emit(event_name, data=data, source="api:manual")
    return {
        "status": "emitted",
        "event_name": event_name,
        "listeners_triggered": log.listeners_triggered,
    }


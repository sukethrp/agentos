from __future__ import annotations
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from agentos.monitor.store import store
from agentos.monitor.ws_manager import get_monitor_manager

router = APIRouter(prefix="/monitor", tags=["monitor"])


def _query(agent_name: str | None, page: int, limit: int) -> tuple[list, int]:
    if agent_name:
        filtered = [e for e in store.events if e.get("agent_name") == agent_name]
    else:
        filtered = store.events
    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    return list(reversed(filtered))[start:end], total


@router.get("/events")
async def monitor_events(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    agent_name: str | None = None,
) -> dict:
    events, total = _query(agent_name, page, limit)
    return {"events": events, "total": total, "page": page, "limit": limit}


@router.get("/stats")
async def monitor_stats() -> dict:
    return store.get_overview()


ws_monitor_router = APIRouter(tags=["monitor"])


@ws_monitor_router.websocket("/monitor")
async def ws_monitor(websocket: WebSocket) -> None:
    mgr = get_monitor_manager()
    await mgr.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        mgr.disconnect(websocket)

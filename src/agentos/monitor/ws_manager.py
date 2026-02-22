from __future__ import annotations
from fastapi import WebSocket
from typing import Any


class MonitorConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        for conn in list(self._connections):
            try:
                await conn.send_json(payload)
            except Exception:
                self.disconnect(conn)


_monitor_manager: MonitorConnectionManager | None = None


def get_monitor_manager() -> MonitorConnectionManager:
    global _monitor_manager
    if _monitor_manager is None:
        _monitor_manager = MonitorConnectionManager()
    return _monitor_manager


async def broadcast_tool_event(agent_id: str, tool_name: str, result: str, latency_ms: float) -> None:
    try:
        mgr = get_monitor_manager()
        await mgr.broadcast({
            "agent_id": agent_id,
            "tool_name": tool_name,
            "result": result[:500] if isinstance(result, str) else str(result)[:500],
            "latency_ms": latency_ms,
        })
    except Exception:
        pass


async def broadcast_team_node_event(team_id: str, node_id: str, status: str, output: str = "") -> None:
    try:
        mgr = get_monitor_manager()
        await mgr.broadcast({
            "team_id": team_id,
            "node_id": node_id,
            "status": status,
            "output": output[:500] if isinstance(output, str) else str(output)[:500],
        })
    except Exception:
        pass


def broadcast_tool_event_sync(agent_id: str, tool_name: str, result: str, latency_ms: float) -> None:
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        asyncio.ensure_future(broadcast_tool_event(agent_id, tool_name, result, latency_ms))
    except RuntimeError:
        asyncio.run(broadcast_tool_event(agent_id, tool_name, result, latency_ms))

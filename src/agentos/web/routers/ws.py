from __future__ import annotations
import asyncio
import queue
import threading
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from agentos.monitor.store import store
from agentos.tools import get_builtin_tools

router = APIRouter(tags=["websocket"])

@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """WebSocket endpoint that streams tokens in real-time."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query", "").strip()
        if not query:
            await websocket.send_json({"type": "error", "message": "Empty query"})
            return

        name = data.get("name", "chat-agent")
        model = data.get("model", "gpt-4o-mini")
        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        tools_list = data.get("tools", [])
        temperature = float(data.get("temperature", 0.7))

        from agentos.core.agent import Agent

        available_tools = get_builtin_tools()
        agent_tools = [available_tools[t] for t in tools_list if t in available_tools]

        agent = Agent(
            name=name,
            model=model,
            tools=agent_tools,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        chunk_queue: queue.Queue = queue.Queue()

        def run_agent_stream():
            try:
                for chunk in agent.run(query, stream=True):
                    chunk_queue.put(("chunk", chunk))
                chunk_queue.put(("done", None))
            except Exception as e:
                chunk_queue.put(("error", str(e)))

        stream_thread = threading.Thread(target=run_agent_stream)
        stream_thread.start()

        full_response = []
        while True:
            try:
                msg_type, data = chunk_queue.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if msg_type == "error":
                await websocket.send_json({"type": "error", "message": data})
                return
            if msg_type == "done":
                break

            if isinstance(data, str):
                full_response.append(data)
                await websocket.send_json({"type": "token", "content": data})

        stream_thread.join(timeout=1.0)

        response_text = "".join(full_response)
        cost = sum(e.cost_usd for e in agent.events)
        tokens = sum(e.tokens_used for e in agent.events)
        tools_used = [
            e.data.get("tool", "") for e in agent.events if e.event_type == "tool_call"
        ]

        for e in agent.events:
            store.log_event(e)

        await websocket.send_json(
            {
                "type": "done",
                "response": response_text,
                "cost": round(cost, 6),
                "tokens": tokens,
                "tools_used": tools_used,
            }
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@router.websocket("/ws/monitor")
async def ws_monitor(websocket: WebSocket):
    from agentos.monitor.ws_manager import get_monitor_manager

    mgr = get_monitor_manager()
    await mgr.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        mgr.disconnect(websocket)

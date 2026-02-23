from __future__ import annotations
import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class MeshMessage(BaseModel):
    sender: str
    receiver: str | None
    topic: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class MeshRouter:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[MeshMessage]] = {}
        self._subscriptions: dict[
            tuple[str, str], list[Callable[[MeshMessage], Awaitable[Any]]]
        ] = {}
        self._delivery_tasks: dict[str, asyncio.Task[None]] = {}
        self._redis = None
        self._redis_task: asyncio.Task[None] | None = None
        redis_url = os.getenv("REDIS_URL")
        self._redis_url = redis_url if redis_url else None

    def _get_queue(self, agent_id: str) -> asyncio.Queue[MeshMessage]:
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
        return self._queues[agent_id]

    def _queue_depth(self, agent_id: str) -> int:
        q = self._queues.get(agent_id)
        return q.qsize() if q else 0

    def registered_agents(self) -> list[str]:
        return list(self._queues.keys())

    def queue_depths(self) -> dict[str, int]:
        return {aid: self._queue_depth(aid) for aid in self._queues}

    async def _delivery_loop(self, agent_id: str) -> None:
        q = self._queues[agent_id]
        while True:
            try:
                msg = await q.get()
                key = (agent_id, msg.topic)
                handlers = self._subscriptions.get(key, [])
                for h in handlers:
                    try:
                        await h(msg)
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break

    def _start_delivery(self, agent_id: str) -> None:
        if agent_id in self._delivery_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
            self._delivery_tasks[agent_id] = loop.create_task(
                self._delivery_loop(agent_id)
            )
        except RuntimeError:
            pass

    async def send_message(
        self,
        from_id: str,
        to_id: str,
        payload: dict[str, Any],
        topic: str = "default",
    ) -> MeshMessage:
        msg = MeshMessage(
            sender=from_id,
            receiver=to_id,
            topic=topic,
            payload=payload,
        )
        q = self._get_queue(to_id)
        await q.put(msg)
        self._start_delivery(to_id)
        if self._redis_url:
            await self._publish_redis(msg)
        return msg

    async def broadcast(
        self,
        from_id: str,
        topic: str,
        payload: dict[str, Any],
    ) -> list[MeshMessage]:
        agents = self.registered_agents()
        msgs: list[MeshMessage] = []
        for aid in agents:
            if aid != from_id:
                msg = await self.send_message(from_id, aid, payload, topic)
                msgs.append(msg)
        if self._redis_url:
            bcast = MeshMessage(
                sender=from_id, receiver=None, topic=topic, payload=payload
            )
            await self._publish_redis(bcast)
            msgs.append(bcast)
        return msgs

    def subscribe(
        self,
        agent_id: str,
        topic: str,
        handler: Callable[[MeshMessage], Awaitable[Any]],
    ) -> None:
        key = (agent_id, topic)
        if key not in self._subscriptions:
            self._subscriptions[key] = []
        self._subscriptions[key].append(handler)
        self._get_queue(agent_id)
        self._start_delivery(agent_id)

    async def _publish_redis(self, msg: MeshMessage) -> None:
        try:
            from redis.asyncio import Redis

            if self._redis is None:
                self._redis = Redis.from_url(self._redis_url, decode_responses=True)
            channel = f"mesh:{msg.topic}"
            await self._redis.publish(channel, msg.model_dump_json())
        except Exception:
            pass

    async def _redis_listener(self) -> None:
        if not self._redis_url:
            return
        try:
            from redis.asyncio import Redis
            import json

            r = Redis.from_url(self._redis_url, decode_responses=True)
            self._redis = r
            pubsub = r.pubsub()
            await pubsub.psubscribe("mesh:*")
            async for m in pubsub.listen():
                if m.get("type") == "pmessage":
                    try:
                        data = json.loads(m.get("data", "{}"))
                        msg = MeshMessage(**data)
                        if msg.receiver:
                            q = self._get_queue(msg.receiver)
                            await q.put(msg)
                            self._start_delivery(msg.receiver)
                        else:
                            for aid in self.registered_agents():
                                if aid != msg.sender:
                                    q = self._get_queue(aid)
                                    await q.put(msg)
                                    self._start_delivery(aid)
                    except Exception:
                        pass
        except Exception:
            pass

    async def start_redis_listener(self) -> None:
        if self._redis_url and self._redis_task is None:
            self._redis_task = asyncio.create_task(self._redis_listener())

    async def stop_redis_listener(self) -> None:
        if self._redis_task:
            self._redis_task.cancel()
            try:
                await self._redis_task
            except asyncio.CancelledError:
                pass
            self._redis_task = None
        if self._redis:
            await self._redis.aclose()
            self._redis = None


_mesh_router: MeshRouter | None = None


def get_mesh_router() -> MeshRouter:
    global _mesh_router
    if _mesh_router is None:
        _mesh_router = MeshRouter()
    return _mesh_router

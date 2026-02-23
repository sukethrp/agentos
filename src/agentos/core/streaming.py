"""StreamingAgent — async streaming wrapper with real-time cost tracking.

Usage:
    from agentos.core.streaming import StreamingAgent

    agent = StreamingAgent(name="streamer", model="gpt-4o-mini")

    # Async streaming (for WebSocket / FastAPI)
    async for token in agent.stream("Tell me a story"):
        print(token, end="", flush=True)
    print(agent.last_cost, agent.last_tokens)

    # Sync streaming (for terminal scripts)
    for token in agent.stream_sync("Tell me a story"):
        print(token, end="", flush=True)

    # Non-streaming still works
    msg = agent.run("Hello")
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Generator
from agentos.core.agent import Agent
from agentos.core.types import AgentEvent, Message
from agentos.core.tool import Tool
from agentos.core.memory import Memory


@dataclass
class StreamStats:
    """Real-time stats tracked during streaming."""

    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tool_calls: int = 0
    llm_calls: int = 0
    first_token_ms: float = 0.0


class StreamingAgent:
    """Wraps Agent with async streaming and real-time cost tracking.

    Provides both async and sync generators for token-by-token output,
    plus live stats that update as tokens arrive.
    """

    def __init__(
        self,
        name: str = "streaming-agent",
        model: str = "gpt-4o-mini",
        tools: list[Tool] | None = None,
        system_prompt: str = "You are a helpful assistant. Use tools when needed.",
        max_iterations: int = 10,
        temperature: float = 0.7,
        memory: Memory | None = None,
    ):
        self.agent = Agent(
            name=name,
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            temperature=temperature,
            memory=memory,
        )
        self.stats = StreamStats()
        self._last_message: Message | None = None

    # ── Async streaming (for WebSocket / FastAPI) ──

    async def stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Async generator that yields tokens as they arrive.

        Usage:
            async for token in agent.stream("Hello"):
                print(token, end="")
        """
        self.stats = StreamStats()
        start = time.time()
        first_token = True

        gen = self.agent.run(user_input, stream=True)

        loop = asyncio.get_event_loop()
        iterator = iter(gen)

        while True:
            try:
                chunk = await loop.run_in_executor(None, next, iterator)
            except StopIteration:
                break

            if isinstance(chunk, str):
                if first_token:
                    self.stats.first_token_ms = (time.time() - start) * 1000
                    first_token = False
                self.stats.tokens += 1
                yield chunk
            elif isinstance(chunk, Message):
                self._last_message = chunk

        self._update_stats_from_events()

    # ── Sync streaming (for terminal / scripts) ──

    def stream_sync(self, user_input: str) -> Generator[str, None, None]:
        """Sync generator that yields tokens as they arrive.

        Usage:
            for token in agent.stream_sync("Hello"):
                print(token, end="", flush=True)
        """
        self.stats = StreamStats()
        start = time.time()
        first_token = True

        for chunk in self.agent.run(user_input, stream=True):
            if isinstance(chunk, str):
                if first_token:
                    self.stats.first_token_ms = (time.time() - start) * 1000
                    first_token = False
                self.stats.tokens += 1
                yield chunk
            elif isinstance(chunk, Message):
                self._last_message = chunk

        self._update_stats_from_events()

    # ── Non-streaming (backward compat) ──

    def run(self, user_input: str) -> Message:
        """Non-streaming run. Same as Agent.run()."""
        msg = self.agent.run(user_input, stream=False)
        self._update_stats_from_events()
        self._last_message = msg
        return msg

    # ── Stats & Properties ──

    @property
    def last_message(self) -> Message | None:
        return self._last_message

    @property
    def last_response(self) -> str:
        """The text content of the last response."""
        if self._last_message and self._last_message.content:
            return self._last_message.content
        return ""

    @property
    def last_cost(self) -> float:
        return self.stats.cost_usd

    @property
    def last_tokens(self) -> int:
        return self.stats.tokens

    @property
    def events(self) -> list[AgentEvent]:
        return self.agent.events

    def _update_stats_from_events(self) -> None:
        """Pull final stats from the agent's event log."""
        self.stats.cost_usd = sum(e.cost_usd for e in self.agent.events)
        self.stats.latency_ms = sum(e.latency_ms for e in self.agent.events)
        self.stats.tool_calls = sum(
            1 for e in self.agent.events if e.event_type == "tool_call"
        )
        self.stats.llm_calls = sum(
            1 for e in self.agent.events if e.event_type == "llm_call"
        )

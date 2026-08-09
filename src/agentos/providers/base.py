from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from agentos.core.tool import Tool
from agentos.core.types import AgentEvent, Message


class BaseProvider(ABC):
    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[Tool],
        model: str,
        temperature: float,
        max_tokens: int,
        agent_name: str,
    ) -> tuple[Message, AgentEvent]:
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        tools: list[Tool],
        model: str,
        temperature: float,
        max_tokens: int,
        agent_name: str,
    ) -> AsyncGenerator[str | tuple[str, Message, AgentEvent], None]:
        pass

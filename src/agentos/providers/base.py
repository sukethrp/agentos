from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from agentos.core.types import Message, AgentEvent
from agentos.core.tool import Tool


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

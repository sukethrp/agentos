"""MCP server that exposes AgentOS tools to any MCP-compatible client."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING

from agentos.core.tool import Tool
from agentos.core.types import ToolCall
from agentos.mcp.adapter import toolspec_to_input_schema

if TYPE_CHECKING:
    from agentos.core.agent import Agent

logger = logging.getLogger(__name__)


def _ensure_mcp():
    """Import and return mcp modules, raising a clear error if not installed."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent
        from mcp.types import Tool as MCPTool
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The 'mcp' package is required for MCP support. "
            "Install it with: pip install 'agentos-platform[mcp]'"
        ) from None
    return Server, stdio_server, MCPTool, TextContent


class MCPServer:
    """Wraps a set of AgentOS Tools as an MCP-compliant server.

    Usage::

        from agentos.mcp import MCPServer
        server = MCPServer("my-server", tools=[calculator, weather])
        server.run()
    """

    def __init__(
        self,
        name: str = "agentos",
        *,
        tools: list[Tool] | None = None,
    ):
        Server, _, MCPTool, TextContent = _ensure_mcp()

        self._name = name
        self._tools: dict[str, Tool] = {}
        self._server = Server(name)
        self._MCPTool = MCPTool
        self._TextContent = TextContent

        if tools:
            for t in tools:
                self.add_tool(t)

        self._register_handlers()

    def add_tool(self, tool: Tool) -> None:
        """Register an AgentOS Tool to be exposed over MCP."""
        self._tools[tool.name] = tool

    @classmethod
    def from_agent(cls, agent: Agent, *, name: str | None = None) -> MCPServer:
        """Create an MCPServer from an existing Agent's registered tools."""
        return cls(
            name=name or f"agentos-{agent.config.name}",
            tools=agent.tools,
        )

    def _register_handlers(self) -> None:
        MCPTool = self._MCPTool
        TextContent = self._TextContent

        @self._server.list_tools()
        async def _list_tools():
            return [
                MCPTool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=toolspec_to_input_schema(tool.spec),
                )
                for tool in self._tools.values()
            ]

        @self._server.call_tool()
        async def _call_tool(name: str, arguments: dict | None = None):
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name}")

            tool = self._tools[name]
            call = ToolCall(name=name, arguments=arguments or {})

            if inspect.iscoroutinefunction(tool.fn):
                result_str = str(await tool.fn(**call.arguments))
            else:
                result_str = (
                    await asyncio.to_thread(tool.execute, call)
                ).result

            logger.debug("MCP tool %s returned %d chars", name, len(result_str))
            return [TextContent(type="text", text=result_str)]

    def run(self) -> None:
        """Start the MCP server on stdio (blocking)."""
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        """Start the MCP server on stdio."""
        _, stdio_server, _, _ = _ensure_mcp()

        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server '%s' starting on stdio", self._name)
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

from __future__ import annotations

import asyncio
import json
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional

from agentos.core.agent import Agent
from agentos.core.tool import Tool
from agentos.core.types import ToolCall, ToolExecutionContext
from agentos.mcp.adapter import toolspec_to_input_schema

_DEFAULT_SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2025-11-25",
    "2025-03-26",
    "2024-11-05",
)


class StdioTransport:
    """Minimal MCP stdio transport.

    MCP stdio transport uses newline-delimited JSON-RPC messages.
    We must never write partial JSON to stdout; each message is serialized
    fully and written in a single write call, then flushed.
    """

    def __init__(
        self,
        stdin: Any = sys.stdin,
        stdout: Any = sys.stdout,
    ) -> None:
        # We purposefully use .buffer to guarantee byte-level atomicity.
        self._stdin = stdin
        self._stdout = stdout
        self._write_lock = threading.Lock()

    def read_message(self) -> Optional[dict[str, Any]]:
        raw = self._stdin.buffer.readline()
        if not raw:
            return None

        # MCP messages are delimited by newlines and must not contain embedded
        # newlines. Clients should therefore send one JSON-RPC message per line.
        line = raw.decode("utf-8").strip()
        if not line:
            return {}

        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("JSON-RPC message must be an object")
        return obj

    def send_message(self, message: dict[str, Any]) -> None:
        # Use compact separators to avoid introducing literal newlines.
        payload = json.dumps(
            message, ensure_ascii=False, separators=(",", ":"), sort_keys=False
        )
        data = (payload + "\n").encode("utf-8")
        # Single atomic write per JSON-RPC message.
        with self._write_lock:
            self._stdout.buffer.write(data)
            self._stdout.flush()


def _jsonrpc_error(
    *,
    id_: Any,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def _jsonrpc_result(*, id_: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


@dataclass(frozen=True)
class _ToolListCursor:
    # Placeholder for future pagination.
    cursor: Any = None


class MCPServer:
    """MCP server exposing AgentOS tools via JSON-RPC over stdio."""

    def __init__(
        self,
        name: str = "agentos",
        *,
        version: str = "0.3.1",
        tools: list[Tool] | None = None,
        supported_protocol_versions: tuple[str, ...] = _DEFAULT_SUPPORTED_PROTOCOL_VERSIONS,
    ) -> None:
        self.name = name
        self.version = version
        self._supported_protocol_versions = supported_protocol_versions
        self._initialized = False

        self._agent: Agent = Agent(
            name=name,
            model="mcp-tools",
            tools=tools or [],
            system_prompt="You are an AgentOS MCP tool server.",
            max_iterations=1,
            temperature=0.0,
        )

    @classmethod
    def from_agent(cls, agent: Agent, *, name: str | None = None) -> "MCPServer":
        # Reuse the agent instance to keep its tool cache/retries behavior.
        server = cls(
            name=name or agent.config.name,
            version="0.3.1",
            tools=list(agent.tools),
        )
        server._agent = agent
        return server

    @property
    def tools(self) -> list[Tool]:
        return list(self._agent.tools)

    def add_tool(self, tool: Tool) -> None:
        # Keep both agent.tools and agent._tool_map consistent.
        if tool.name in self._agent._tool_map:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._agent.tools.append(tool)
        self._agent._tool_map[tool.name] = tool

    def _negotiate_protocol_version(self, requested: Any) -> str:
        if isinstance(requested, str) and requested in self._supported_protocol_versions:
            return requested
        return self._supported_protocol_versions[0]

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested_version = params.get("protocolVersion")
        negotiated = self._negotiate_protocol_version(requested_version)

        # Capability negotiation: we only advertise what we actually implement.
        capabilities: dict[str, Any] = {
            "tools": {"listChanged": False}
        }

        return {
            "protocolVersion": negotiated,
            "capabilities": capabilities,
            "serverInfo": {
                "name": self.name,
                "title": self.name,
                "version": self.version,
                "description": f"AgentOS MCP server exposing {len(self._agent.tools)} tools.",
            },
            "instructions": (
                "You can discover available tools via tools/list and invoke them "
                "via tools/call."
            ),
        }

    def _handle_tools_list(self, params: dict[str, Any] | None) -> dict[str, Any]:
        # cursor is currently ignored; tool list is small and static.
        cursor = (params or {}).get("cursor")
        _ = _ToolListCursor(cursor=cursor)  # keep for future pagination

        tools_out: list[dict[str, Any]] = []
        for tool in self._agent.tools:
            tools_out.append(
                {
                    "name": tool.name,
                    "title": tool.name,
                    "description": tool.description,
                    "inputSchema": toolspec_to_input_schema(tool.spec),
                }
            )

        return {"tools": tools_out, "nextCursor": None}

    def _execute_tool_call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        tool = self._agent._tool_map.get(name)
        if not tool:
            raise KeyError(name)

        tc = ToolCall(name=name, arguments=arguments)
        session_id = getattr(self._agent, "_session_id", "")
        ctx = ToolExecutionContext(
            agent_id=self.name,
            session_id=session_id,
            budget_remaining=0.0,
        )
        (result_str, _latency_ms) = self._agent._execute_tools_batch([tc], ctx)[0]
        is_error = isinstance(result_str, str) and result_str.startswith("ERROR:")
        return result_str, is_error

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise ValueError("Missing or invalid 'name'")
        if not isinstance(arguments, dict):
            raise ValueError("'arguments' must be an object")

        result_str, is_error = self._execute_tool_call(name=name, arguments=arguments)
        return {
            "content": [{"type": "text", "text": result_str}],
            "isError": is_error,
        }

    def _dispatch_message(self, msg: dict[str, Any]) -> Optional[dict[str, Any]]:
        # Requests have a non-null id; notifications omit id entirely.
        is_request = ("id" in msg) and (msg.get("id") is not None)
        id_ = msg.get("id")

        method = msg.get("method")
        if not isinstance(method, str) or not method:
            return (
                _jsonrpc_error(
                    id_=id_,
                    code=-32600,
                    message="Invalid Request",
                    data={"reason": "Missing method"},
                )
                if is_request
                else None
            )

        params = msg.get("params") or {}
        if params is None:
            params = {}
        if not isinstance(params, dict):
            # JSON-RPC requires params to be an object here for our usage.
            return (
                _jsonrpc_error(
                    id_=id_,
                    code=-32602,
                    message="Invalid params",
                    data={"reason": "params must be an object"},
                )
                if is_request
                else None
            )

        if method == "initialize":
            if not is_request:
                # Invalid JSON-RPC; initialize must be a request.
                return _jsonrpc_error(
                    id_=None,
                    code=-32600,
                    message="Invalid Request",
                    data={"reason": "initialize must be a request"},
                )
            if not isinstance(params, dict):
                return _jsonrpc_error(
                    id_=id_,
                    code=-32602,
                    message="Invalid params",
                )
            result = self._handle_initialize(params)
            return _jsonrpc_result(id_=id_, result=result)

        if method == "tools/list":
            if not is_request:
                # tools/list is a request.
                return _jsonrpc_error(
                    id_=None,
                    code=-32600,
                    message="Invalid Request",
                    data={"reason": "tools/list must be a request"},
                )
            result = self._handle_tools_list(params)
            return _jsonrpc_result(id_=id_, result=result)

        if method == "tools/call":
            if not is_request:
                return _jsonrpc_error(
                    id_=None,
                    code=-32600,
                    message="Invalid Request",
                    data={"reason": "tools/call must be a request"},
                )
            try:
                result = self._handle_tools_call(params)
            except KeyError:
                return _jsonrpc_error(
                    id_=id_,
                    code=-32602,
                    message="Unknown tool",
                    data={"tool": params.get("name")},
                )
            except ValueError as e:
                return _jsonrpc_error(
                    id_=id_,
                    code=-32602,
                    message="Invalid params",
                    data={"reason": str(e)},
                )
            except Exception as e:
                return _jsonrpc_error(
                    id_=id_,
                    code=-32603,
                    message="Internal error",
                    data={"reason": str(e)},
                )

            return _jsonrpc_result(id_=id_, result=result)

        if method == "notifications/initialized":
            self._initialized = True
            return None

        # Unknown method
        return (
            _jsonrpc_error(
                id_=id_,
                code=-32601,
                message="Method not found",
                data={"method": method},
            )
            if is_request
            else None
        )

    def run(self) -> None:
        transport = StdioTransport()
        while True:
            try:
                msg = transport.read_message()
            except json.JSONDecodeError:
                # Parse errors should be responded as JSON-RPC errors when possible.
                # Since we don't have an id, use id=null.
                transport.send_message(
                    _jsonrpc_error(
                        id_=None,
                        code=-32700,
                        message="Parse error",
                    )
                )
                continue
            except Exception as e:
                transport.send_message(
                    _jsonrpc_error(
                        id_=None,
                        code=-32603,
                        message="Internal error",
                        data={"reason": str(e)},
                    )
                )
                continue

            if msg is None:
                return
            if not msg:
                continue

            try:
                response = self._dispatch_message(msg)
                if response is not None:
                    transport.send_message(response)
            except Exception as e:
                # Last-resort: ensure we still reply with an atomic JSON-RPC message.
                id_ = msg.get("id") if isinstance(msg, dict) else None
                transport.send_message(
                    _jsonrpc_error(
                        id_=id_,
                        code=-32603,
                        message="Internal error",
                        data={"reason": str(e)},
                    )
                )

    async def run_async(self) -> None:
        await asyncio.to_thread(self.run)


__all__ = ["MCPServer", "StdioTransport"]


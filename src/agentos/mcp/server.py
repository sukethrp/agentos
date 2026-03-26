from __future__ import annotations

import asyncio
import json
import sys
import uuid
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from agentos.core.agent import Agent
from agentos.core.tool import Tool
from agentos.core.types import ToolCall, ToolExecutionContext
from agentos.mcp.adapter import toolspec_to_input_schema

try:
    # SSE transport uses FastAPI/Starlette.
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    import uvicorn
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]

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


class MCPProtocolHandler:
    """Transport-agnostic protocol handler.

    Both stdio and SSE transports delegate JSON-RPC method dispatch to this
    handler so tool discovery/execution logic stays consistent.
    """

    def __init__(self, server: "MCPServer") -> None:
        self._server = server

    def dispatch_sync(
        self, msg: dict[str, Any], *, session_id: str
    ) -> Optional[dict[str, Any]]:
        return self._server._dispatch_message(msg, session_id=session_id)

    async def dispatch_async(
        self, msg: dict[str, Any], *, session_id: str
    ) -> Optional[dict[str, Any]]:
        # Tool execution uses asyncio.run internally in Agent, so we call it
        # from a worker thread to avoid "asyncio.run() cannot be called from
        # a running event loop" errors.
        return await asyncio.to_thread(
            self.dispatch_sync,
            msg,
            session_id=session_id,
        )


class _SseSessionState:
    session_id: str
    subscribers: set[asyncio.Queue[tuple[str, dict[str, Any]]]]
    history: list[tuple[str, dict[str, Any]]]
    next_event_id: int = 1
    active_tasks: set[asyncio.Task[Any]]

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.subscribers = set()
        self.history = []
        self.active_tasks = set()


class SseTransport:
    """SSE transport for MCP JSON-RPC messages."""

    def __init__(
        self,
        *,
        protocol: MCPProtocolHandler,
        host: str = "127.0.0.1",
        port: int = 8080,
        sse_path: str = "/sse",
        messages_path: str = "/messages",
        max_history: int = 2000,
    ) -> None:
        if FastAPI is None or uvicorn is None:
            raise RuntimeError(
                "SSE transport requires FastAPI/uvicorn dependencies."
            )

        self._protocol = protocol
        self._host = host
        self._port = port
        self._sse_path = sse_path
        self._messages_path = messages_path
        self._max_history = max_history

        self._sessions: dict[str, _SseSessionState] = {}
        self._sessions_lock = asyncio.Lock()

    async def _get_session(self, session_id: str) -> _SseSessionState:
        async with self._sessions_lock:
            s = self._sessions.get(session_id)
            if s is None:
                s = _SseSessionState(session_id=session_id)
                self._sessions[session_id] = s
            return s

    async def _enqueue(self, session_id: str, message: dict[str, Any]) -> None:
        s = await self._get_session(session_id)
        event_id = str(s.next_event_id)
        s.next_event_id += 1

        s.history.append((event_id, message))
        if len(s.history) > self._max_history:
            s.history = s.history[-self._max_history :]

        # Broadcast to all active SSE subscribers.
        for q in list(s.subscribers):
            try:
                q.put_nowait((event_id, message))
            except Exception:
                pass

    async def _cancel_tasks(self, session_id: str) -> None:
        s = await self._get_session(session_id)
        tasks = list(s.active_tasks)
        s.active_tasks.clear()
        for t in tasks:
            t.cancel()

    def _format_event(self, event_id: str, payload: dict[str, Any]) -> str:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"id: {event_id}\ndata: {data}\n\n"

    async def _sse_event_stream(
        self,
        *,
        request: "Request",
        session_id: str,
    ) -> AsyncIterator[str]:
        s = await self._get_session(session_id)

        subscriber_q: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        s.subscribers.add(subscriber_q)

        last_event_id = request.headers.get("Last-Event-ID")
        last_num = -1
        if last_event_id is not None:
            try:
                last_num = int(last_event_id)
            except ValueError:
                last_num = -1

        try:
            # Replay buffered history (resumability).
            for ev_id, msg in list(s.history):
                try:
                    if int(ev_id) > last_num:
                        yield self._format_event(ev_id, msg)
                except Exception:
                    yield self._format_event(ev_id, msg)

            # Stream new events.
            while True:
                ev_id, msg = await subscriber_q.get()
                yield self._format_event(ev_id, msg)
        except asyncio.CancelledError:
            raise
        finally:
            # Handle disconnect.
            try:
                s.subscribers.discard(subscriber_q)
            except Exception:
                pass

            # If this was the last client connection for the session, cancel
            # any in-flight tool executions for that session.
            if not s.subscribers:
                await self._cancel_tasks(session_id)

    def run(self) -> None:
        app = FastAPI(title="AgentOS MCP SSE Transport", version="0.3.1")

        @app.get(self._sse_path)
        async def sse_endpoint(request: "Request") -> StreamingResponse:
            session_id = request.headers.get("MCP-Session-Id")
            if not session_id:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Missing MCP-Session-Id header for SSE subscription"
                    },
                )

            stream = self._sse_event_stream(
                request=request,
                session_id=session_id,
            )
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        @app.post(self._messages_path)
        async def messages_endpoint(request: "Request") -> Response:
            body = await request.json()
            if not isinstance(body, dict):
                return JSONResponse(status_code=400, content={"error": "Invalid body"})

            method = body.get("method")
            if method == "initialize":
                session_id = request.headers.get("MCP-Session-Id") or uuid.uuid4().hex
                await self._get_session(session_id)

                # Initialization must return a proper JSON-RPC response.
                resp = self._protocol.dispatch_sync(body, session_id=session_id)
                headers = {"MCP-Session-Id": session_id}
                return JSONResponse(
                    status_code=200 if resp is not None else 204,
                    content=resp if resp is not None else {},
                    headers=headers,
                )

            session_id = request.headers.get("MCP-Session-Id")
            if not session_id:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Missing MCP-Session-Id header for non-initialize requests"
                    },
                )

            async def handle_one() -> None:
                s = await self._get_session(session_id)
                task = asyncio.current_task()
                if task is not None:
                    s.active_tasks.add(task)
                try:
                    resp = await self._protocol.dispatch_async(
                        body, session_id=session_id
                    )
                    if resp is not None:
                        await self._enqueue(session_id, resp)
                except asyncio.CancelledError:
                    # Client disconnected; do not send results.
                    raise
                except Exception as e:
                    err = _jsonrpc_error(
                        id_=body.get("id"),
                        code=-32603,
                        message="Internal error",
                        data={"reason": str(e)},
                    )
                    await self._enqueue(session_id, err)
                finally:
                    s = await self._get_session(session_id)
                    if task is not None:
                        s.active_tasks.discard(task)

            asyncio.create_task(handle_one())
            return Response(status_code=202)

        uvicorn.run(app, host=self._host, port=self._port)


class MCPServer:
    """MCP server exposing AgentOS tools via JSON-RPC over stdio."""

    def __init__(
        self,
        name: str = "agentos",
        *,
        version: str = "0.3.1",
        tools: list[Tool] | None = None,
        transport: str = "stdio",
        sse_host: str = "127.0.0.1",
        sse_port: int = 8080,
        sse_path: str = "/sse",
        messages_path: str = "/messages",
        supported_protocol_versions: tuple[str, ...] = _DEFAULT_SUPPORTED_PROTOCOL_VERSIONS,
    ) -> None:
        self.name = name
        self.version = version
        self._supported_protocol_versions = supported_protocol_versions
        self._initialized = False
        self._initialized_sessions: set[str] = set()
        self._transport = transport

        self._agent: Agent = Agent(
            name=name,
            model="mcp-tools",
            tools=tools or [],
            system_prompt="You are an AgentOS MCP tool server.",
            max_iterations=1,
            temperature=0.0,
        )

        self._protocol = MCPProtocolHandler(self)
        self._sse_transport: SseTransport | None = None
        if transport == "sse":
            self._sse_transport = SseTransport(
                protocol=self._protocol,
                host=sse_host,
                port=sse_port,
                sse_path=sse_path,
                messages_path=messages_path,
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

    def _execute_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> tuple[str, bool]:
        tool = self._agent._tool_map.get(name)
        if not tool:
            raise KeyError(name)

        tc = ToolCall(name=name, arguments=arguments)
        session_id = session_id or getattr(self._agent, "_session_id", "")
        ctx = ToolExecutionContext(
            agent_id=self.name,
            session_id=session_id,
            budget_remaining=0.0,
        )
        (result_str, _latency_ms) = self._agent._execute_tools_batch([tc], ctx)[0]
        is_error = isinstance(result_str, str) and result_str.startswith("ERROR:")
        return result_str, is_error

    def _handle_tools_call(
        self, params: dict[str, Any], *, session_id: str
    ) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise ValueError("Missing or invalid 'name'")
        if not isinstance(arguments, dict):
            raise ValueError("'arguments' must be an object")

        result_str, is_error = self._execute_tool_call(
            name=name, arguments=arguments, session_id=session_id
        )
        return {
            "content": [{"type": "text", "text": result_str}],
            "isError": is_error,
        }

    def _dispatch_message(
        self, msg: dict[str, Any], *, session_id: str
    ) -> Optional[dict[str, Any]]:
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
                result = self._handle_tools_call(params, session_id=session_id)
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
            self._initialized_sessions.add(session_id)
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
        if self._transport == "stdio":
            transport = StdioTransport()
            while True:
                try:
                    msg = transport.read_message()
                except json.JSONDecodeError:
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
                    response = self._protocol.dispatch_sync(
                        msg,
                        session_id="stdio",
                    )
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
        elif self._transport == "sse":
            if not self._sse_transport:
                raise RuntimeError("SSE transport is not configured")
            self._sse_transport.run()
        else:
            raise ValueError(f"Unknown transport: {self._transport}")

    async def run_async(self) -> None:
        await asyncio.to_thread(self.run)


__all__ = ["MCPServer", "StdioTransport", "SseTransport", "MCPProtocolHandler"]


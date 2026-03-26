from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pytest
import textwrap


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for(condition, *, timeout_s: float = 10.0, interval_s: float = 0.1) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            if condition():
                return
        except BaseException as e:  # pragma: no cover
            last_err = e
        time.sleep(interval_s)
    raise TimeoutError(f"Condition not met within {timeout_s}s: {last_err}")


def _write_json_line(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    fp.flush()


def _read_json_line(stdout, *, timeout_s: float = 10.0) -> dict[str, Any]:
    q: "queue.Queue[str]" = queue.Queue()
    err_q: "queue.Queue[BaseException]" = queue.Queue()

    def _reader() -> None:
        try:
            line = stdout.readline()
            if line is None:
                return
            q.put(line)
        except BaseException as e:  # pragma: no cover
            err_q.put(e)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        line = q.get(timeout=timeout_s)
    except queue.Empty:
        raise TimeoutError("Timed out waiting for JSON-RPC message from stdout")
    except BaseException as e:
        raise e

    buf = line.strip()
    if buf == "":
        raise RuntimeError(
            "Received empty line from MCP server stdout (stdout closed unexpectedly)."
        )
    # Buffering correctness check: stdout messages are line-delimited and
    # each line must be a complete JSON-RPC object.
    return json.loads(buf)


def _make_agent_module(tmp_path: Path, *, tool_variants: str) -> Path:
    """
    Create a temporary python module file defining an Agent named `agent`.

    `tool_variants` should contain tool definitions and any supporting code.
    """
    module = tmp_path / "agent.py"
    module.write_text(
        f"""
from agentos.core.agent import Agent
from agentos.core.tool import tool

{textwrap.dedent(tool_variants)}

agent = Agent(
    name="mcp-test-agent",
    model="mcp-test-model",
    tools=[add, boom, echo_no_args],
)
""".lstrip(),
        encoding="utf-8",
    )
    return module.parent


def _spawn_stdio_server(
    *, repo_root: Path, agent_dir: Path, name: str
) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "agentos.cli",
        "mcp",
        "serve",
        "--transport",
        "stdio",
        "--name",
        name,
        "--agent",
        str(agent_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    return proc


def _send_initialize(*, request_id: int = 1) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {},
        },
    }


def _stdio_tool_call(
    *, request_id: int, tool_name: str, args: dict[str, Any]
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": args},
    }


def _stdio_tools_list(*, request_id: int) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/list", "params": {}}


@pytest.mark.parametrize("transport", ["stdio"])
def test_mcp_stdio_roundtrip_and_buffering(
    repo_root: Path, tmp_path: Path, transport: str
) -> None:
    agent_dir = _make_agent_module(
        tmp_path,
        tool_variants="""
@tool(description="Add two numbers")
def add(a: int, b: int) -> str:
    return str(a + b)


@tool(description="Always fails")
def boom(x: str) -> str:
    raise RuntimeError("boom called")


@tool(description="No args")
def echo_no_args() -> str:
    return "no-args-ok"
""",
    )

    proc = _spawn_stdio_server(
        repo_root=repo_root, agent_dir=agent_dir, name="mcp-stdio-test"
    )
    try:
        _write_json_line(proc.stdin, _send_initialize())
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr is not None else ""
            raise AssertionError(
                f"MCP stdio server exited early. stderr tail:\n{err[-2000:]}"
            )
        try:
            init = _read_json_line(proc.stdout)
        except Exception as e:
            if proc.poll() is not None and proc.stderr is not None:
                err = proc.stderr.read()
                raise AssertionError(
                    f"MCP stdio server failed while reading initialize. stderr tail:\n{err[-2000:]}"
                ) from e
            raise
        assert init["result"]["protocolVersion"] == "2025-11-25"

        _write_json_line(proc.stdin, _stdio_tools_list(request_id=2))
        tools_list = _read_json_line(proc.stdout)
        tools = tools_list["result"]["tools"]
        tool_names = {t["name"] for t in tools}
        assert {"add", "boom", "echo_no_args"} <= tool_names

        _write_json_line(
            proc.stdin,
            _stdio_tool_call(request_id=3, tool_name="add", args={"a": 2, "b": 5}),
        )
        call_res = _read_json_line(proc.stdout)
        assert call_res["id"] == 3
        assert call_res["result"]["isError"] is False
        assert call_res["result"]["content"][0]["type"] == "text"
        assert call_res["result"]["content"][0]["text"] == "7"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_stdio_error_handling(repo_root: Path, tmp_path: Path) -> None:
    agent_dir = _make_agent_module(
        tmp_path,
        tool_variants="""
@tool(description="Add two numbers")
def add(a: int, b: int) -> str:
    return str(a + b)

@tool(description="Always fails")
def boom(x: str) -> str:
    raise RuntimeError("boom called")

@tool(description="No args")
def echo_no_args() -> str:
    return "no-args-ok"
""",
    )

    proc = _spawn_stdio_server(
        repo_root=repo_root, agent_dir=agent_dir, name="mcp-stdio-errors"
    )
    try:
        _write_json_line(proc.stdin, _send_initialize())
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr is not None else ""
            raise AssertionError(
                f"MCP stdio server exited early. stderr tail:\n{err[-2000:]}"
            )
        try:
            _ = _read_json_line(proc.stdout)
        except Exception as e:
            if proc.poll() is not None and proc.stderr is not None:
                err = proc.stderr.read()
                raise AssertionError(
                    f"MCP stdio server failed while reading initialize. stderr tail:\n{err[-2000:]}"
                ) from e
            raise

        # Invalid JSON-RPC payload (parse error)
        proc.stdin.write("not-json\n")
        proc.stdin.flush()
        parse_err = _read_json_line(proc.stdout)
        assert parse_err["error"]["code"] == -32700

        # Unknown method
        _write_json_line(
            proc.stdin,
            {"jsonrpc": "2.0", "id": 10, "method": "tools/unknown", "params": {}},
        )
        unknown_err = _read_json_line(proc.stdout)
        assert unknown_err["error"]["code"] == -32601
        assert unknown_err["error"]["data"]["method"] == "tools/unknown"

        # Tool execution failure
        _write_json_line(
            proc.stdin,
            _stdio_tool_call(request_id=12, tool_name="boom", args={"x": "hi"}),
        )
        boom_res = _read_json_line(proc.stdout)
        assert boom_res["id"] == 12
        assert boom_res["result"]["isError"] is True
        assert boom_res["result"]["content"][0]["text"].startswith("ERROR:")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_stdio_tool_discovery_schemas(repo_root: Path, tmp_path: Path) -> None:
    agent_dir = _make_agent_module(
        tmp_path,
        tool_variants="""
@tool(description="Add two numbers")
def add(a: int, b: int = 3) -> str:
    return str(a + b)

@tool(description="Always fails")
def boom(x: str) -> str:
    raise RuntimeError("boom called")

@tool(description="No args")
def echo_no_args() -> str:
    return "no-args-ok"
""",
    )

    proc = _spawn_stdio_server(
        repo_root=repo_root, agent_dir=agent_dir, name="mcp-stdio-discovery"
    )
    try:
        _write_json_line(proc.stdin, _send_initialize())
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr is not None else ""
            raise AssertionError(
                f"MCP stdio server exited early. stderr tail:\n{err[-2000:]}"
            )
        try:
            _ = _read_json_line(proc.stdout)
        except Exception as e:
            if proc.poll() is not None and proc.stderr is not None:
                err = proc.stderr.read()
                raise AssertionError(
                    f"MCP stdio server failed while reading initialize. stderr tail:\n{err[-2000:]}"
                ) from e
            raise

        _write_json_line(proc.stdin, _stdio_tools_list(request_id=2))
        tools_list = _read_json_line(proc.stdout)
        tools = tools_list["result"]["tools"]
        by_name = {t["name"]: t for t in tools}

        add_schema = by_name["add"]["inputSchema"]
        assert add_schema["type"] == "object"
        assert "properties" in add_schema
        assert set(add_schema["properties"].keys()) == {"a", "b"}
        # `b` has a default => not required
        assert set(add_schema["required"]) == {"a"}
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_sse_roundtrip(repo_root: Path, tmp_path: Path) -> None:
    try:
        import httpx
    except Exception:
        pytest.skip("httpx not installed")

    agent_dir = _make_agent_module(
        tmp_path,
        tool_variants="""
@tool(description="Add two numbers")
def add(a: int, b: int) -> str:
    return str(a + b)

@tool(description="Always fails")
def boom(x: str) -> str:
    raise RuntimeError("boom called")

@tool(description="No args")
def echo_no_args() -> str:
    return "no-args-ok"
""",
    )

    port = _get_free_port()
    host = "127.0.0.1"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agentos.cli",
            "mcp",
            "serve",
            "--transport",
            "sse",
            "--host",
            host,
            "--port",
            str(port),
            "--name",
            "mcp-sse-test",
            "--agent",
            str(agent_dir),
        ],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    client = httpx.Client(timeout=10.0)
    stop_reader = threading.Event()
    t = None
    try:
        # Wait until server is reachable by calling /messages with initialize.
        def _init_ok() -> bool:
            try:
                r = client.post(
                    f"http://{host}:{port}/messages",
                    json=_send_initialize(request_id=1),
                )
                return r.status_code == 200
            except Exception:
                return False

        _wait_for(_init_ok, timeout_s=20.0)

        init_resp = client.post(
            f"http://{host}:{port}/messages",
            json=_send_initialize(request_id=1),
            headers={},
        )
        assert init_resp.status_code == 200
        session_id = init_resp.headers.get("MCP-Session-Id")
        assert session_id
        init_body = init_resp.json()
        assert init_body["result"]["protocolVersion"] == "2025-11-25"

        sse_headers = {"MCP-Session-Id": session_id}
        sse_url = f"http://{host}:{port}/sse"

        sse_queue: "queue.Queue[tuple[int, dict[str, Any]]]" = queue.Queue()

        def sse_reader() -> None:
            # Use a separate streaming connection to avoid interfering with
            # request/response calls.
            stream_client = httpx.Client(timeout=None)
            try:
                with stream_client.stream("GET", sse_url, headers=sse_headers) as r:
                    assert r.status_code == 200
                    event_lines: list[str] = []
                    try:
                        for line in r.iter_lines():
                            if stop_reader.is_set():
                                return
                            if line is None:
                                continue
                            if line == "":
                                if not event_lines:
                                    continue
                                event_id: Optional[str] = None
                                data_str: Optional[str] = None
                                for el in event_lines:
                                    if el.startswith("id: "):
                                        event_id = el[len("id: ") :]
                                    elif el.startswith("data: "):
                                        data_str = el[len("data: ") :]
                                event_lines = []
                                if data_str is None or event_id is None:
                                    continue
                                sse_queue.put((int(event_id), json.loads(data_str)))
                                continue
                            event_lines.append(line)
                    except Exception:
                        # When the test tears down, the server process is killed,
                        # which can cause incomplete chunked reads. Ignore.
                        return
            finally:
                try:
                    stream_client.close()
                except Exception:
                    pass

        t = threading.Thread(target=sse_reader, daemon=True)
        t.start()

        # Send tools/list; response should be delivered over SSE.
        list_req = _stdio_tools_list(request_id=2)
        list_post = client.post(
            f"http://{host}:{port}/messages",
            json=list_req,
            headers=sse_headers,
        )
        assert list_post.status_code == 202

        req_id_to_find = 2
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                _ev_id, msg = sse_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg.get("id") == req_id_to_find:
                tools = msg["result"]["tools"]
                tool_names = {t["name"] for t in tools}
                assert {"add", "boom", "echo_no_args"} <= tool_names
                break
        else:
            raise AssertionError("Did not receive tools/list response over SSE")

        # Send tools/call; response should be delivered over SSE.
        call_req = _stdio_tool_call(
            request_id=3, tool_name="add", args={"a": 2, "b": 5}
        )
        call_post = client.post(
            f"http://{host}:{port}/messages",
            json=call_req,
            headers=sse_headers,
        )
        assert call_post.status_code == 202

        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                _ev_id, msg = sse_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg.get("id") == 3:
                assert msg["result"]["isError"] is False
                assert msg["result"]["content"][0]["text"] == "7"
                break
        else:
            raise AssertionError("Did not receive tools/call response over SSE")
    finally:
        stop_reader.set()
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

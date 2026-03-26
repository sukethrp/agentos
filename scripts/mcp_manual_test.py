from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _claude_config_path() -> Path:
    # Claude Desktop config path differs by platform.
    if sys.platform.startswith("darwin"):
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    # Linux / fallback
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def _build_claude_snippet(name: str, transport: str, *, agent: str | None) -> str:
    args = ["mcp", "serve", "--transport", transport, "--name", name]
    if agent:
        args += ["--agent", agent]

    snippet = {
        "mcpServers": {
            name: {
                "command": "agentos",
                "args": args,
            }
        }
    }
    import json

    return json.dumps(snippet, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start an AgentOS MCP server and print Claude Desktop config snippet."
    )
    parser.add_argument("--name", default="agentos-mcp", help="MCP server name")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport to run",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host")
    parser.add_argument("--port", type=int, default=8080, help="SSE port")
    parser.add_argument(
        "--agent",
        default=None,
        help="Optional agent module/file/directory to load (same as `agentos mcp serve --agent`)",
    )

    args = parser.parse_args()

    config_path = _claude_config_path()
    snippet = _build_claude_snippet(args.name, args.transport, agent=args.agent)

    print(f"Claude Desktop config path:\n  {config_path}")
    print("\nAdd this snippet to your `claude_desktop_config.json`:\n")
    print(snippet)
    print("\nStarting MCP server now...\n")

    # Use the installed `agentos` CLI entrypoint for consistency.
    cmd = [
        sys.executable,
        "-m",
        "agentos.cli",
        "mcp",
        "serve",
        "--transport",
        args.transport,
        "--name",
        args.name,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.agent:
        cmd += ["--agent", args.agent]

    # Replace current process with the server process.
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()


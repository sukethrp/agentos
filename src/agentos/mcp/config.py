from __future__ import annotations

from typing import Any


def generate_mcp_config(
    *,
    name: str = "agentos",
    command: str = "python",
    args: list[str] | None = None,
    env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a Cursor/Claude Desktop style MCP server config snippet.

    This returns a structure compatible with the common `mcpServers` schema:

    {
      "mcpServers": {
        "<name>": { "command": "...", "args": [...] }
      }
    }
    """

    spec: dict[str, Any] = {
        "command": command,
        "args": args or [],
    }
    if env:
        # Some clients accept `env`; others ignore it. We include it when provided.
        spec["env"] = env

    return {"mcpServers": {name: spec}}


__all__ = ["generate_mcp_config"]


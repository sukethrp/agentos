"""MCP (Model Context Protocol) Server for AgentOS.

Exposes AgentOS tools as MCP-compatible endpoints, allowing any
MCP client (Claude Desktop, Cursor, etc.) to discover and use
AgentOS agent tools.

Design decision (see docs/adr/006-mcp-support.md):
We implement the MCP server protocol directly rather than depending
on the official SDK to keep the dependency footprint minimal.
The protocol is simple enough (JSON-RPC over stdio) that a direct
implementation is cleaner for our use case.
"""

from agentos.mcp.config import generate_mcp_config
from agentos.mcp.server import MCPServer

__all__ = ["MCPServer", "generate_mcp_config"]

# MCP (Model Context Protocol) Support

AgentOS can expose agent tools as **MCP-compatible endpoints**, allowing any
MCP client — Claude Desktop, Cursor, or custom integrations — to discover and
call your tools.

## Quick Start

### 1. Install the MCP extra

```bash
pip install 'agentos-platform[mcp]'
```

### 2. Define tools and start the server

```python
from agentos.core.tool import tool
from agentos.mcp import MCPServer


@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    return str(eval(expression))


server = MCPServer("my-server", tools=[calculator])
server.run()  # starts on stdio
```

### 3. Configure your MCP client

Add the server to your client's MCP config (e.g. Claude Desktop
`claude_desktop_config.json` or Cursor `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "agentos": {
      "command": "python",
      "args": ["path/to/your_server.py"]
    }
  }
}
```

## Creating an MCP Server from an Agent

If you already have an `Agent` with tools, you can expose it directly:

```python
from agentos.core.agent import Agent
from agentos.core.tool import tool


@tool(description="Get weather for a city")
def weather(city: str) -> str:
    return "72°F, Sunny"


agent = Agent(
    name="assistant",
    model="gpt-4o-mini",
    tools=[weather],
)

server = agent.as_mcp_server()
server.run()
```

## API Reference

### `MCPServer`

```python
MCPServer(name: str = "agentos", *, tools: list[Tool] | None = None)
```

| Method | Description |
|--------|-------------|
| `add_tool(tool)` | Register an additional AgentOS `Tool` |
| `from_agent(agent, *, name=None)` | Class method — create server from an `Agent`'s tools |
| `run()` | Start the server on stdio (blocking) |
| `run_async()` | Async version of `run()` |

### `Agent.as_mcp_server(name=None)`

Convenience method that returns an `MCPServer` wrapping the agent's tools.

## Architecture

```
┌──────────────┐      stdio / SSE       ┌──────────────────┐
│  MCP Client  │  ◄──────────────────►  │  AgentOS MCP     │
│  (Cursor,    │     MCP Protocol       │  Server          │
│   Claude)    │                        │                  │
└──────────────┘                        │  ┌────────────┐  │
                                        │  │ @tool fn() │  │
                                        │  │ @tool fn() │  │
                                        │  │ @tool fn() │  │
                                        │  └────────────┘  │
                                        └──────────────────┘
```

The adapter layer converts between AgentOS types and MCP types:

- **`ToolSpec`** → MCP `Tool` (name, description, inputSchema)
- **`ToolCall` / `ToolResult`** → MCP `CallToolRequest` / `TextContent`

## References

- [Model Context Protocol specification](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

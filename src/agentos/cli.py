"""
AgentOS CLI — command-line interface for managing agents.

Usage:
    agentos serve          Start the web platform on port 8000
    agentos serve --port 3000   Start on custom port
    agentos serve --demo   Start in demo mode (no API keys needed)
    agentos test           Run agent test scenarios
    agentos mcp serve      Start MCP server for Claude Desktop/Cursor
    agentos version        Show version
    agentos init           Create a new agent project scaffold
"""

import argparse
import os
import importlib
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="agentos",
        description="AgentOS — The Operating System for AI Agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # agentos serve
    serve_parser = subparsers.add_parser("serve", help="Start web platform")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--demo", action="store_true",
                              help="Run in demo mode without API keys")

    # agentos mcp
    mcp_parser = subparsers.add_parser("mcp", help="Model Context Protocol commands")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", required=True)

    mcp_serve_parser = mcp_subparsers.add_parser("serve", help="Start MCP server")
    mcp_serve_parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport to use",
    )
    mcp_serve_parser.add_argument("--host", default="127.0.0.1")
    mcp_serve_parser.add_argument("--port", type=int, default=8080)
    mcp_serve_parser.add_argument("--name", default="agentos")
    mcp_serve_parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help=(
            "Agent name or Python path. If you pass a path, it should "
            "point to a .py file (or a directory containing agent.py). "
            "The module should expose `agent` or `AGENT`."
        ),
    )

    # agentos version
    subparsers.add_parser("version", help="Show version")

    # agentos init
    init_parser = subparsers.add_parser("init", help="Create new agent project")
    init_parser.add_argument("name", nargs="?", default="my-agent")

    args = parser.parse_args()

    if args.command == "serve":
        if args.demo:
            os.environ["AGENTOS_DEMO_MODE"] = "true"
        import uvicorn
        from agentos.web.app import app
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "mcp" and args.mcp_command == "serve":
        from agentos.core.agent import Agent
        from agentos.mcp import MCPServer
        from agentos.tools import get_builtin_tools

        def _load_agent(agent_ref: str) -> Agent:
            p = Path(agent_ref).expanduser()
            module = None
            if p.exists():
                if p.is_dir():
                    p = p / "agent.py"
                if not p.exists():
                    raise FileNotFoundError(f"Agent file not found: {p}")
                spec = importlib.util.spec_from_file_location(
                    f"agentos_user_agent_{p.stem}_{os.getpid()}",
                    str(p),
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load module from: {p}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(agent_ref)

            obj = getattr(module, "agent", None) or getattr(module, "AGENT", None)
            if obj is None and hasattr(module, "get_agent"):
                obj = module.get_agent()
            if obj is None:
                raise AttributeError(
                    f"Agent module must define `agent` or `AGENT` (or `get_agent()`): {agent_ref}"
                )

            # Support GovernedAgent-style wrappers (has `.agent` attribute).
            inner = getattr(obj, "agent", None)
            if isinstance(inner, Agent):
                return inner
            if isinstance(obj, Agent):
                return obj
            raise TypeError(
                "Loaded object is not an agentos Agent. "
                "Expected an `agent`/`AGENT` instance (AgentOS Agent), "
                "or a GovernedAgent wrapper with `.agent`."
            )

        if args.agent:
            agent_obj = _load_agent(args.agent)
            server = MCPServer(
                name=args.name,
                tools=list(agent_obj.tools),
                transport=args.transport,
                sse_host=args.host,
                sse_port=args.port,
            )
        else:
            tools_dict = get_builtin_tools()
            tools = list(tools_dict.values())
            server = MCPServer(
                name=args.name,
                tools=tools,
                transport=args.transport,
                sse_host=args.host,
                sse_port=args.port,
            )

        print(
            f"MCP server ready (name={server.name}, transport={args.transport})"
        )
        server.run()

    elif args.command == "version":
        from agentos import __version__
        print(f"AgentOS v{__version__}")

    elif args.command == "init":
        _init_project(args.name)

    else:
        parser.print_help()


def _init_project(name: str):
    """Scaffold a new AgentOS agent project."""
    os.makedirs(name, exist_ok=True)

    agent_code = f'''from agentos.governed_agent import GovernedAgent
from agentos.core.tool import tool
from agentos.governance.budget import BudgetGuard


@tool(description="Describe what this tool does")
def my_tool(input: str) -> str:
    return f"Processed: {{input}}"


agent = GovernedAgent(
    name="{name}",
    model="gpt-4o-mini",
    tools=[my_tool],
    budget=BudgetGuard(max_per_day=5.00),
)

if __name__ == "__main__":
    result = agent.run("Hello!")
    print(result.content)
'''

    with open(os.path.join(name, "agent.py"), "w") as f:
        f.write(agent_code)

    with open(os.path.join(name, ".env"), "w") as f:
        f.write("OPENAI_API_KEY=sk-your-key-here\n")

    print(f"Created agent project: {name}/")
    print(f"   cd {name}")
    print("   # Add your API key to .env")
    print("   python agent.py")


if __name__ == "__main__":
    main()

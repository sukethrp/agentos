"""
AgentOS CLI — command-line interface for managing agents.

Usage:
    agentos serve          Start the web platform on port 8000
    agentos serve --port 3000   Start on custom port
    agentos serve --demo   Start in demo mode (no API keys needed)
    agentos test           Run agent test scenarios
    agentos mcp            Start MCP server for Claude Desktop/Cursor
    agentos version        Show version
    agentos init           Create a new agent project scaffold
"""

import argparse
import os


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
    subparsers.add_parser("mcp", help="Start MCP server")

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

    elif args.command == "mcp":
        from agentos.mcp.adapter import toolspec_to_input_schema  # noqa: F401
        from agentos.tools import get_builtin_tools
        tools = get_builtin_tools()
        print(f"MCP server ready with {len(tools)} tools")
        print("Note: full MCP server requires the 'mcp' extra — pip install agentos-platform[mcp]")

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

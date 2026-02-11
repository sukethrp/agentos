"""AgentOS Plugin Demo — discover, load, and use plugins.

Shows:
  1. Discover plugins in the plugins/ directory
  2. Load all plugins
  3. List available plugins and tools
  4. Create an agent with plugin-provided tools
  5. Run queries using those tools
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentos.plugins import PluginManager
from agentos.core.agent import Agent
from agentos.tools import get_builtin_tools


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    divider("🔌 AgentOS — Plugin System Demo")

    plugins_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")

    # ── Step 1: Discover plugins ──
    pm = PluginManager()
    discovered = pm.discover_plugins(plugins_dir)
    print(f"📂 Discovered {len(discovered)} plugin(s) in {plugins_dir}:")
    for name in discovered:
        info = pm.get_plugin(name)
        print(f"   • {name} ({info.path})")

    # ── Step 2: Load all plugins ──
    divider("Step 2: Load Plugins")
    results = pm.load_all()
    for name, success in results.items():
        info = pm.get_plugin(name)
        if success:
            inst = info.instance
            print(f"   ✅ {inst.name} v{inst.version} — {inst.description or 'no description'}")
        else:
            print(f"   ❌ {name} — {info.error}")

    # ── Step 3: List available tools ──
    divider("Step 3: Available Tools")
    plugin_tools = pm.get_tools()
    print(f"   Plugin tools ({len(plugin_tools)}):")
    for name, tool in plugin_tools.items():
        desc = getattr(tool, "description", "")[:70]
        print(f"   • {name}: {desc}")

    builtin_tools = get_builtin_tools()
    print(f"\n   Built-in tools ({len(builtin_tools)}):")
    for name, tool in builtin_tools.items():
        desc = getattr(tool, "description", "")[:70]
        print(f"   • {name}: {desc}")

    # ── Step 4: Overview ──
    divider("Step 4: Plugin Manager Overview")
    overview = pm.get_overview()
    print(f"   Total plugins:   {overview['total_plugins']}")
    print(f"   Loaded plugins:  {overview['loaded_plugins']}")
    print(f"   Plugin tools:    {overview['total_tools']}")
    print(f"   Plugin providers: {overview['total_providers']}")

    # ── Step 5: Create agent with plugin tools ──
    divider("Step 5: Agent with Plugin Tools")

    # Combine built-in + plugin tools
    all_tools = list(builtin_tools.values()) + pm.get_tools_list()
    print(f"   Total tools for agent: {len(all_tools)}")
    print(f"   Tools: {', '.join(t.name for t in all_tools)}")

    agent = Agent(
        name="plugin-agent",
        model="gpt-4o-mini",
        tools=all_tools,
        system_prompt=(
            "You are a helpful assistant with access to various tools including "
            "a translator and GitHub tools. Use the translate tool to translate "
            "phrases. Be concise."
        ),
    )

    # ── Step 6: Test the translate plugin ──
    divider("Step 6: Test Translate Plugin")

    queries = [
        "Translate 'hello' to Japanese using the translate tool",
        "How do you say 'thank you' in French? Use the translate tool.",
    ]

    for q in queries:
        print(f"   🗣️ Query: {q}")
        try:
            result = agent.run(q)
            content = result.content or "(no response)"
            cost = sum(e.cost_usd for e in agent.events)
            tokens = sum(e.tokens_used for e in agent.events)
            print(f"   🤖 Agent: {content[:200]}")
            print(f"      💰 ${cost:.4f} | 🎫 {tokens} tokens")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()

    # ── Step 7: Test GitHub plugin (if GITHUB_TOKEN set) ──
    divider("Step 7: Test GitHub Plugin")

    if os.getenv("GITHUB_TOKEN"):
        print("   🔑 GITHUB_TOKEN found — testing GitHub tools...")
        try:
            result = agent.run("List my GitHub repositories using the list_repos tool")
            print(f"   🤖 Agent: {result.content[:300]}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   ⚠️  GITHUB_TOKEN not set — skipping GitHub tool test.")
        print("   Set GITHUB_TOKEN in .env to test GitHub integration.")
        print()
        print("   Testing list_repos directly (public repos, no auth)...")
        # Direct tool test without auth
        list_repos_tool = plugin_tools.get("list_repos")
        if list_repos_tool:
            try:
                output = list_repos_tool.fn(username="python")
                print(f"   {output[:300]}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    # ── Summary ──
    divider("Summary")
    print(f"   Plugins loaded:  {overview['loaded_plugins']}/{overview['total_plugins']}")
    print(f"   Tools available: {len(all_tools)} (built-in: {len(builtin_tools)}, plugins: {len(plugin_tools)})")
    print()
    print(f"{'=' * 60}")
    print(f"  ✅ Plugin demo complete")
    print(f"{'=' * 60}")

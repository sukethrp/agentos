"""BasePlugin — abstract base class for all AgentOS plugins.

A plugin can register tools, providers, or middleware with AgentOS.
Plugins inherit from BasePlugin and implement on_load() to set things up.

Example:

    class MyPlugin(BasePlugin):
        name = "my-plugin"
        version = "0.3.0"
        description = "Adds cool tools"
        author = "me"

        def on_load(self, ctx):
            self.register_tool(ctx, my_tool)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PluginContext:
    """Passed to plugins during registration.

    Plugins use this to add tools, providers, and middleware
    to the running AgentOS instance.
    """

    tools: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Callable] = field(default_factory=dict)
    middleware: list[Callable] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def add_tool(self, name: str, tool: Any) -> None:
        """Register a tool by name."""
        self.tools[name] = tool

    def add_provider(self, name: str, provider_fn: Callable) -> None:
        """Register a model provider by name."""
        self.providers[name] = provider_fn

    def add_middleware(self, fn: Callable) -> None:
        """Register middleware that wraps agent execution."""
        self.middleware.append(fn)


class BasePlugin:
    """Base class for AgentOS plugins.

    Subclass this and implement on_load() to register your tools,
    providers, or middleware.
    """

    name: str = "unnamed-plugin"
    version: str = "0.3.0"
    description: str = ""
    author: str = ""

    def __init__(self) -> None:
        self._loaded = False

    def on_load(self, ctx: PluginContext) -> None:
        """Called when the plugin is loaded. Override to register tools/providers."""
        pass

    def on_unload(self) -> None:
        """Called when the plugin is unloaded. Override for cleanup."""
        pass

    def register_tool(self, ctx: PluginContext, tool: Any) -> None:
        """Convenience: register a Tool instance on the context."""
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        ctx.add_tool(tool_name, tool)

    def register_provider(self, ctx: PluginContext, name: str, provider_fn: Callable) -> None:
        """Convenience: register a model provider on the context."""
        ctx.add_provider(name, provider_fn)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "loaded": self._loaded,
        }

    def __repr__(self) -> str:
        return f"<Plugin {self.name} v{self.version}>"

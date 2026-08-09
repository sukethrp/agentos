"""AgentOS Plugin System — extensible tools, providers, and features."""

from agentos.plugins.base import BasePlugin, PluginContext
from agentos.plugins.manager import PluginInfo, PluginManager

__all__ = [
    "BasePlugin",
    "PluginContext",
    "PluginInfo",
    "PluginManager",
]

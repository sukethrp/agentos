"""PluginManager — discover, load, and manage AgentOS plugins.

Usage:
    from agentos.plugins import PluginManager

    pm = PluginManager()
    pm.discover_plugins("plugins/")   # scan directory
    pm.load_all()                     # load everything found
    tools = pm.get_tools()            # dict[str, Tool]

Plugins are Python files that contain either:
  - A ``register(ctx)`` function (simple style), OR
  - A subclass of ``BasePlugin`` (class style).

Both styles receive a PluginContext to register tools, providers, etc.
"""

from __future__ import annotations
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentos.plugins.base import BasePlugin, PluginContext


@dataclass
class PluginInfo:
    """Metadata about a discovered (but not necessarily loaded) plugin."""

    name: str
    path: str
    plugin_type: str = "unknown"  # "class" or "function"
    loaded: bool = False
    instance: BasePlugin | None = None
    error: str = ""

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "path": self.path,
            "type": self.plugin_type,
            "loaded": self.loaded,
        }
        if self.instance:
            d.update(self.instance.to_dict())
        if self.error:
            d["error"] = self.error
        return d


class PluginManager:
    """Discovers, loads, and manages plugins for AgentOS.

    Args:
        auto_load: If True, discovered plugins are loaded immediately.
    """

    def __init__(self, auto_load: bool = False):
        self._plugins: dict[str, PluginInfo] = {}
        self._ctx = PluginContext()
        self._auto_load = auto_load

    # ── Discovery ──

    def discover_plugins(self, directory: str) -> list[str]:
        """Scan a directory for plugin files (*.py, not __init__).

        Returns list of discovered plugin names.
        """
        directory = str(Path(directory).resolve())
        if not os.path.isdir(directory):
            return []

        discovered: list[str] = []

        for entry in sorted(os.scandir(directory), key=lambda e: e.name):
            if not entry.is_file() or not entry.name.endswith(".py"):
                continue
            if entry.name.startswith("_"):
                continue

            plugin_name = entry.name[:-3]  # strip .py
            self._plugins[plugin_name] = PluginInfo(
                name=plugin_name,
                path=entry.path,
            )
            discovered.append(plugin_name)

            if self._auto_load:
                self.load_plugin(plugin_name)

        return discovered

    # ── Loading ──

    def load_plugin(self, name: str) -> bool:
        """Load and initialize a single plugin by name.

        Returns True on success.
        """
        info = self._plugins.get(name)
        if not info:
            return False
        if info.loaded:
            return True

        try:
            module = self._import_file(name, info.path)
        except Exception as e:
            info.error = f"Import failed: {e}"
            return False

        # Strategy 1: look for a BasePlugin subclass
        plugin_cls = self._find_plugin_class(module)
        if plugin_cls:
            return self._load_class_plugin(info, plugin_cls)

        # Strategy 2: look for a register(ctx) function
        register_fn = getattr(module, "register", None)
        if callable(register_fn):
            return self._load_function_plugin(info, register_fn)

        info.error = "No BasePlugin subclass or register() function found"
        return False

    def load_all(self) -> dict[str, bool]:
        """Load all discovered plugins. Returns {name: success}."""
        results = {}
        for name in list(self._plugins):
            results[name] = self.load_plugin(name)
        return results

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin (calls on_unload if available)."""
        info = self._plugins.get(name)
        if not info or not info.loaded:
            return False

        if info.instance:
            try:
                info.instance.on_unload()
            except Exception:
                pass

        # Remove tools registered by this plugin
        plugin_prefix = f"{name}:"
        to_remove = [k for k in self._ctx.tools if k.startswith(plugin_prefix) or k == name]
        for k in to_remove:
            del self._ctx.tools[k]

        info.loaded = False
        info.instance = None
        return True

    # ── Query ──

    def list_plugins(self) -> list[PluginInfo]:
        """Return info about all discovered plugins."""
        return list(self._plugins.values())

    def get_plugin(self, name: str) -> PluginInfo | None:
        return self._plugins.get(name)

    def get_tools(self) -> dict[str, Any]:
        """Return all tools registered by all loaded plugins."""
        return dict(self._ctx.tools)

    def get_tools_list(self) -> list[Any]:
        """Return tools as a flat list (ready to pass to Agent)."""
        return list(self._ctx.tools.values())

    def get_providers(self) -> dict[str, Any]:
        """Return all providers registered by plugins."""
        return dict(self._ctx.providers)

    @property
    def context(self) -> PluginContext:
        """The shared plugin context."""
        return self._ctx

    def get_overview(self) -> dict:
        total = len(self._plugins)
        loaded = sum(1 for p in self._plugins.values() if p.loaded)
        return {
            "total_plugins": total,
            "loaded_plugins": loaded,
            "total_tools": len(self._ctx.tools),
            "total_providers": len(self._ctx.providers),
            "plugins": [p.to_dict() for p in self._plugins.values()],
        }

    # ── Internal helpers ──

    @staticmethod
    def _import_file(name: str, path: str):
        """Import a Python file as a module."""
        spec = importlib.util.spec_from_file_location(f"agentos_plugin_{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _find_plugin_class(module) -> type | None:
        """Find the first BasePlugin subclass in a module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BasePlugin)
                and attr is not BasePlugin
            ):
                return attr
        return None

    def _load_class_plugin(self, info: PluginInfo, cls: type) -> bool:
        """Instantiate a BasePlugin subclass and call on_load."""
        try:
            instance = cls()
            instance.on_load(self._ctx)
            instance._loaded = True

            info.plugin_type = "class"
            info.loaded = True
            info.instance = instance
            return True
        except Exception as e:
            info.error = f"on_load failed: {e}"
            return False

    def _load_function_plugin(self, info: PluginInfo, register_fn) -> bool:
        """Call a register(ctx) function."""
        try:
            register_fn(self._ctx)

            # Create a synthetic BasePlugin to hold metadata
            instance = BasePlugin()
            # Try to read metadata from the module
            module = sys.modules.get(f"agentos_plugin_{info.name}")
            if module:
                instance.name = getattr(module, "PLUGIN_NAME", info.name)
                instance.version = getattr(module, "PLUGIN_VERSION", "0.3.0")
                instance.description = getattr(module, "PLUGIN_DESCRIPTION", "")
                instance.author = getattr(module, "PLUGIN_AUTHOR", "")
            else:
                instance.name = info.name
            instance._loaded = True

            info.plugin_type = "function"
            info.loaded = True
            info.instance = instance
            return True
        except Exception as e:
            info.error = f"register() failed: {e}"
            return False

from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from agentos.marketplace.manifest import PackageManifest

_ROOT = Path.home() / ".agentos"
_REGISTRY_PATH = _ROOT / "marketplace.json"
_PACKAGES_DIR = _ROOT / "packages"


def _ensure_root() -> None:
    _ROOT.mkdir(parents=True, exist_ok=True)
    _PACKAGES_DIR.mkdir(parents=True, exist_ok=True)


class MarketplaceRegistry:
    def __init__(self, registry_path: Path | str | None = None):
        self._path = Path(registry_path) if registry_path else _REGISTRY_PATH
        self._packages: dict[str, dict] = {}
        _ensure_root()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._packages = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._packages = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._packages, indent=2))

    def _key(self, name: str, version: str) -> str:
        return f"{name}@{version}"

    def list_packages(self) -> list[dict]:
        return list(self._packages.values())

    def get(self, name: str, version: str | None = None) -> dict | None:
        if version:
            return self._packages.get(self._key(name, version))
        matches = [v for k, v in self._packages.items() if v.get("name") == name]
        if not matches:
            return None
        def _ver_key(x):
            v = x.get("version", "0")
            parts = []
            for p in v.split(".")[:3]:
                try:
                    parts.append(int(p))
                except ValueError:
                    parts.append(0)
            return tuple(parts)
        return max(matches, key=_ver_key)

    def search(self, tags: str = "", capability: str = "") -> list[dict]:
        results = list(self._packages.values())
        if tags:
            tag_set = {t.strip().lower() for t in tags.split(",") if t.strip()}
            results = [p for p in results if tag_set & {t.lower() for t in p.get("tags", [])}]
        if capability:
            cap = capability.lower()
            results = [p for p in results if cap in (p.get("capability") or "").lower()]
        return results


def publish(manifest_path: str) -> PackageManifest:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    raw = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    manifest = PackageManifest.model_validate(data)
    registry = MarketplaceRegistry()
    pkg_dir = _PACKAGES_DIR / f"{manifest.name}-{manifest.version}"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    src_dir = path.parent
    for item in src_dir.iterdir():
        if item.name in (".git", "__pycache__", ".venv", "venv"):
            continue
        dst = pkg_dir / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    entry = {
        "name": manifest.name,
        "version": manifest.version,
        "description": manifest.description,
        "author": manifest.author,
        "entry_point": manifest.entry_point,
        "inputs": manifest.inputs,
        "outputs": manifest.outputs,
        "pricing_model": manifest.pricing_model,
        "tags": manifest.tags,
        "capability": manifest.capability,
        "path": str(pkg_dir),
    }
    registry._packages[registry._key(manifest.name, manifest.version)] = entry
    registry._save()
    return manifest


def install(name: str, version: str | None = None) -> dict | None:
    import sys
    import importlib
    registry = MarketplaceRegistry()
    pkg = registry.get(name, version)
    if not pkg:
        return None
    pkg_path = pkg.get("path", "")
    if not pkg_path or not Path(pkg_path).exists():
        return pkg
    entry_point = pkg.get("entry_point", "")
    if not entry_point:
        return pkg
    try:
        module_path, attr = entry_point.split(":", 1)
        pkg_root = str(Path(pkg_path).resolve())
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
        mod = importlib.import_module(module_path)
        obj = getattr(mod, attr)
        from agentos.marketplace.registry import _installed_tools
        from agentos.core.tool import Tool
        resolved = obj() if callable(obj) else obj
        tool_name = resolved.name if isinstance(resolved, Tool) else name
        _installed_tools[tool_name] = resolved
        return pkg
    except Exception:
        return pkg


_installed_tools: dict = {}

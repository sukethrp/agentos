"""Mesh Discovery — agent registry and DNS-like lookup.

The :class:`MeshRegistry` is an in-memory phone-book where agents
register their :class:`MeshIdentity`.  Other agents can search by
mesh_id, capability, or organisation.

In a production deployment the registry would be backed by a real
service (etcd, Consul, or a shared HTTP registry).  For local / demo
use, a global singleton is provided.
"""

from __future__ import annotations

import fnmatch
import time
from typing import Any

from agentos.mesh.protocol import MeshIdentity


class MeshRegistry:
    """In-memory agent registry with search capabilities."""

    def __init__(self) -> None:
        self._agents: dict[str, MeshIdentity] = {}
        self._heartbeats: dict[str, float] = {}

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, identity: MeshIdentity) -> None:
        """Register or update an agent's identity in the registry."""
        self._agents[identity.mesh_id] = identity
        self._heartbeats[identity.mesh_id] = time.time()

    def deregister(self, mesh_id: str) -> bool:
        if mesh_id in self._agents:
            del self._agents[mesh_id]
            self._heartbeats.pop(mesh_id, None)
            return True
        return False

    def heartbeat(self, mesh_id: str) -> None:
        """Update the last-seen timestamp for an agent."""
        if mesh_id in self._agents:
            self._heartbeats[mesh_id] = time.time()

    # ── Lookup ───────────────────────────────────────────────────────────

    def lookup(self, mesh_id: str) -> MeshIdentity | None:
        """Exact lookup by mesh_id (like DNS A-record)."""
        return self._agents.get(mesh_id)

    def resolve(self, pattern: str) -> list[MeshIdentity]:
        """Glob-style lookup — e.g. ``*@acme.com`` or ``sales-*``."""
        return [
            a for mid, a in self._agents.items()
            if fnmatch.fnmatch(mid, pattern)
        ]

    def search(
        self,
        query: str = "",
        capability: str = "",
        organisation: str = "",
    ) -> list[MeshIdentity]:
        """Multi-field search across the registry."""
        results = list(self._agents.values())

        if organisation:
            org = organisation.lower()
            results = [a for a in results if a.organisation.lower() == org]

        if capability:
            cap = capability.lower()
            results = [
                a for a in results
                if any(cap in c.lower() for c in a.capabilities)
            ]

        if query:
            q = query.lower()
            results = [
                a for a in results
                if q in a.mesh_id.lower()
                or q in a.display_name.lower()
                or q in a.organisation.lower()
                or any(q in c.lower() for c in a.capabilities)
            ]

        return results

    # ── Listing ──────────────────────────────────────────────────────────

    def list_all(self) -> list[MeshIdentity]:
        return list(self._agents.values())

    def list_online(self, timeout_seconds: float = 120) -> list[MeshIdentity]:
        """Return agents whose heartbeat is within *timeout_seconds*."""
        cutoff = time.time() - timeout_seconds
        return [
            self._agents[mid]
            for mid, ts in self._heartbeats.items()
            if ts >= cutoff and mid in self._agents
        ]

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "total_agents": len(self._agents),
            "organisations": sorted({a.organisation for a in self._agents.values() if a.organisation}),
            "capabilities": sorted({c for a in self._agents.values() for c in a.capabilities}),
        }

    def to_list(self) -> list[dict]:
        """Serialise the registry for API responses."""
        now = time.time()
        out = []
        for mid, identity in self._agents.items():
            d = identity.model_dump()
            d["last_seen"] = self._heartbeats.get(mid, 0)
            d["online"] = (now - self._heartbeats.get(mid, 0)) < 120
            out.append(d)
        return out


# ── Default global registry ──────────────────────────────────────────────────

_default_registry: MeshRegistry | None = None


def get_registry() -> MeshRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = MeshRegistry()
    return _default_registry

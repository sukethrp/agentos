"""Persistent Storage — Agents save and retrieve data across sessions.

Supports multiple backends: JSON file, SQLite, or in-memory.

Usage:
    store = AgentStorage("my-agent", backend="json", path="./data")
    store.set("customer_list", [{"name": "Acme", "value": "$50K"}])
    store.set("last_report", "Q4 revenue up 15%...")

    # Later (even after restart):
    customers = store.get("customer_list")
    report = store.get("last_report")

    # List all keys
    store.list_keys()

    # Search
    results = store.search("customer")
"""

from __future__ import annotations
import json
import os
import time
import sqlite3
from typing import Any


class AgentStorage:
    """Persistent key-value storage for agents."""

    def __init__(self, agent_name: str, backend: str = "json", path: str = "./agent_data"):
        self.agent_name = agent_name
        self.backend = backend
        self.path = path
        self._data: dict[str, dict] = {}

        if backend == "json":
            self._init_json()
        elif backend == "sqlite":
            self._init_sqlite()
        elif backend == "memory":
            pass
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'json', 'sqlite', or 'memory'.")

    # ── JSON Backend ──

    def _json_path(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        return os.path.join(self.path, f"{self.agent_name}.json")

    def _init_json(self):
        fp = self._json_path()
        if os.path.exists(fp):
            with open(fp, "r") as f:
                self._data = json.load(f)

    def _save_json(self):
        fp = self._json_path()
        with open(fp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    # ── SQLite Backend ──

    def _sqlite_path(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        return os.path.join(self.path, f"{self.agent_name}.db")

    def _init_sqlite(self):
        conn = sqlite3.connect(self._sqlite_path())
        conn.execute("""
            CREATE TABLE IF NOT EXISTS store (
                key TEXT PRIMARY KEY,
                value TEXT,
                type TEXT,
                updated_at REAL
            )
        """)
        conn.commit()
        conn.close()

    # ── Core Operations ──

    def set(self, key: str, value: Any, metadata: dict | None = None):
        """Store a value."""
        entry = {
            "value": value,
            "type": type(value).__name__,
            "updated_at": time.time(),
            "metadata": metadata or {},
        }

        if self.backend == "sqlite":
            conn = sqlite3.connect(self._sqlite_path())
            conn.execute(
                "INSERT OR REPLACE INTO store (key, value, type, updated_at) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value, default=str), type(value).__name__, time.time()),
            )
            conn.commit()
            conn.close()
        else:
            self._data[key] = entry
            if self.backend == "json":
                self._save_json()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value."""
        if self.backend == "sqlite":
            conn = sqlite3.connect(self._sqlite_path())
            cur = conn.execute("SELECT value, type FROM store WHERE key = ?", (key,))
            row = cur.fetchone()
            conn.close()
            if row:
                return json.loads(row[0])
            return default
        else:
            entry = self._data.get(key)
            return entry["value"] if entry else default

    def delete(self, key: str) -> bool:
        """Delete a key."""
        if self.backend == "sqlite":
            conn = sqlite3.connect(self._sqlite_path())
            conn.execute("DELETE FROM store WHERE key = ?", (key,))
            conn.commit()
            conn.close()
            return True
        else:
            if key in self._data:
                del self._data[key]
                if self.backend == "json":
                    self._save_json()
                return True
            return False

    def list_keys(self) -> list[str]:
        """List all stored keys."""
        if self.backend == "sqlite":
            conn = sqlite3.connect(self._sqlite_path())
            cur = conn.execute("SELECT key FROM store ORDER BY key")
            keys = [row[0] for row in cur.fetchall()]
            conn.close()
            return keys
        else:
            return list(self._data.keys())

    def search(self, query: str) -> list[dict]:
        """Search keys and values containing the query string."""
        results = []
        query_lower = query.lower()

        if self.backend == "sqlite":
            conn = sqlite3.connect(self._sqlite_path())
            cur = conn.execute(
                "SELECT key, value FROM store WHERE key LIKE ? OR value LIKE ?",
                (f"%{query}%", f"%{query}%"),
            )
            for row in cur.fetchall():
                results.append({"key": row[0], "value": json.loads(row[1])})
            conn.close()
        else:
            for key, entry in self._data.items():
                val_str = json.dumps(entry["value"], default=str).lower()
                if query_lower in key.lower() or query_lower in val_str:
                    results.append({"key": key, "value": entry["value"]})

        return results

    def get_all(self) -> dict[str, Any]:
        """Get all stored data."""
        if self.backend == "sqlite":
            conn = sqlite3.connect(self._sqlite_path())
            cur = conn.execute("SELECT key, value FROM store")
            data = {row[0]: json.loads(row[1]) for row in cur.fetchall()}
            conn.close()
            return data
        else:
            return {k: v["value"] for k, v in self._data.items()}

    def clear(self):
        """Delete all data."""
        if self.backend == "sqlite":
            conn = sqlite3.connect(self._sqlite_path())
            conn.execute("DELETE FROM store")
            conn.commit()
            conn.close()
        else:
            self._data = {}
            if self.backend == "json":
                self._save_json()

    def stats(self) -> dict:
        """Get storage statistics."""
        keys = self.list_keys()
        return {
            "agent": self.agent_name,
            "backend": self.backend,
            "total_keys": len(keys),
            "keys": keys[:20],
        }

    def print_status(self):
        s = self.stats()
        print(f"\n{'='*60}")
        print(f"💾 Storage: {s['agent']} ({s['backend']})")
        print(f"{'='*60}")
        print(f"   Keys stored: {s['total_keys']}")
        if s['keys']:
            print(f"   Keys: {', '.join(s['keys'][:10])}")
        print(f"{'='*60}")
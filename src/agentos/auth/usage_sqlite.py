from __future__ import annotations
import asyncio
import sqlite3
from pathlib import Path
from dataclasses import dataclass


@dataclass
class UsageSummary:
    api_key: str
    queries: int
    tokens: int
    cost: float

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key,
            "queries": self.queries,
            "tokens": self.tokens,
            "cost": self.cost,
        }


def _db_path() -> Path:
    return Path(__file__).parent / "usage.db"


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key_hash TEXT NOT NULL,
            org_id TEXT,
            tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0,
            month TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_api_key ON usage_records(api_key_hash, month)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_org ON usage_records(org_id, month)"
    )
    conn.commit()


def _hash_key(key: str) -> str:
    import hashlib

    return hashlib.sha256(key.encode()).hexdigest()


class UsageTrackerAsync:
    def __init__(self, db_path: str | Path | None = None):
        self._path = Path(db_path) if db_path else _db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        _init_db(conn)
        conn.close()

    async def record(
        self, api_key: str, tokens: int, cost: float, org_id: str | None = None
    ) -> None:
        if org_id is None:
            try:
                from agentos.auth.org_store import get_api_key_info

                info = get_api_key_info(api_key)
                if info:
                    org_id = info.org_id
            except Exception:
                pass
        import time

        now = time.time()
        ts = time.gmtime(now)
        month = f"{ts.tm_year}-{ts.tm_mon:02d}"
        kh = _hash_key(api_key)

        def _insert():
            conn = sqlite3.connect(self._path)
            conn.execute(
                "INSERT INTO usage_records (api_key_hash, org_id, tokens, cost, month, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (kh, org_id, tokens, cost, month, now),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_insert)

    async def get_usage(self, api_key: str, month: str) -> UsageSummary:
        kh = _hash_key(api_key)

        def _query():
            conn = sqlite3.connect(self._path)
            cur = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens), 0), COALESCE(SUM(cost), 0) FROM usage_records WHERE api_key_hash = ? AND month = ?",
                (kh, month),
            )
            row = cur.fetchone()
            conn.close()
            return row or (0, 0, 0.0)

        count, tokens, cost = await asyncio.to_thread(_query)
        return UsageSummary(
            api_key=api_key, queries=count, tokens=int(tokens), cost=float(cost)
        )

    async def get_org_usage(self, org_id: str, month: str) -> tuple[int, float]:
        def _query():
            conn = sqlite3.connect(self._path)
            cur = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0), COALESCE(SUM(cost), 0) FROM usage_records WHERE org_id = ? AND month = ?",
                (org_id, month),
            )
            row = cur.fetchone()
            conn.close()
            return row or (0, 0.0)

        tokens, cost = await asyncio.to_thread(_query)
        return int(tokens), float(cost)

    def get_org_usage_sync(self, org_id: str, month: str) -> tuple[int, float]:
        conn = sqlite3.connect(self._path)
        cur = conn.execute(
            "SELECT COALESCE(SUM(tokens), 0), COALESCE(SUM(cost), 0) FROM usage_records WHERE org_id = ? AND month = ?",
            (org_id, month),
        )
        row = cur.fetchone()
        conn.close()
        return (int(row[0]), float(row[1])) if row else (0, 0.0)


usage_tracker_async = UsageTrackerAsync()

"""Database Query Tool — run read-only SQL queries against a SQLite database.

Only SELECT statements are allowed to prevent accidental data modification.
"""

from __future__ import annotations

import json
import sqlite3

from agentos.core.tool import Tool


def database_query_tool(
    db_path: str = ":memory:",
    *,
    max_rows: int = 50,
) -> Tool:
    """Create a tool that runs read-only SQL queries against a SQLite database.

    Args:
        db_path: Path to the SQLite file, or ``:memory:`` for an in-memory DB.
        max_rows: Maximum number of rows to return per query.
    """

    def query_database(sql: str) -> str:
        """Execute a read-only SQL query and return results as JSON."""
        normalized = sql.strip().rstrip(";").upper()
        if not normalized.startswith("SELECT"):
            return "Error: Only SELECT queries are allowed for safety."

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(max_rows)
            if not rows:
                return "Query returned 0 rows."
            columns = rows[0].keys()
            results = [dict(zip(columns, row)) for row in rows]
            return json.dumps(results, indent=2, default=str)
        except sqlite3.Error as e:
            return f"SQL Error: {e}"
        finally:
            conn.close()

    return Tool(
        fn=query_database,
        name="database_query",
        description=(
            "Run a read-only SQL SELECT query against a SQLite database "
            "and return results as JSON. Only SELECT statements are permitted."
        ),
    )

"""JSON / CSV Parser Tool — parse structured data files and answer questions.

Handles JSON files (objects, arrays) and CSV files, returning summaries
or specific fields that an LLM can reason over.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from agentos.core.tool import Tool


def json_parser_tool(*, max_chars: int = 4000) -> Tool:
    """Create a tool that parses a JSON file or string and returns a summary."""

    def parse_json(source: str) -> str:
        """Parse JSON from a file path or raw string. Returns a formatted summary."""
        try:
            if Path(source).is_file():
                raw = Path(source).read_text(encoding="utf-8")
            else:
                raw = source

            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            return f"Error parsing JSON: {e}"

        if isinstance(data, list):
            summary = (
                f"JSON array with {len(data)} items.\n"
                f"First item keys: {list(data[0].keys()) if data and isinstance(data[0], dict) else 'N/A'}\n"
            )
            preview = json.dumps(data[:5], indent=2, default=str)
        elif isinstance(data, dict):
            summary = f"JSON object with keys: {list(data.keys())}\n"
            preview = json.dumps(data, indent=2, default=str)
        else:
            summary = f"JSON value: {type(data).__name__}\n"
            preview = json.dumps(data, default=str)

        result = summary + "\n" + preview
        if len(result) > max_chars:
            result = result[:max_chars] + "..."
        return result

    return Tool(
        fn=parse_json,
        name="parse_json",
        description=(
            "Parse a JSON file or raw JSON string. Returns a summary of "
            "the structure (keys, array length) and a preview of the data. "
            "Provide a file path or JSON text."
        ),
    )


def csv_parser_tool(*, max_rows: int = 20) -> Tool:
    """Create a tool that parses a CSV file or string and returns a summary."""

    def parse_csv(source: str) -> str:
        """Parse CSV from a file path or raw string. Returns column info and sample rows."""
        try:
            if Path(source).is_file():
                raw = Path(source).read_text(encoding="utf-8")
            else:
                raw = source

            reader = csv.DictReader(io.StringIO(raw))
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(dict(row))
        except Exception as e:
            return f"Error parsing CSV: {e}"

        if not rows:
            return "CSV is empty or could not be parsed."

        columns = list(rows[0].keys())
        total_hint = f" (showing first {max_rows})" if len(rows) == max_rows else ""
        summary = (
            f"CSV with {len(columns)} columns: {columns}\n"
            f"{len(rows)} rows{total_hint}\n\n"
        )
        preview = json.dumps(rows[:10], indent=2, default=str)
        return summary + preview

    return Tool(
        fn=parse_csv,
        name="parse_csv",
        description=(
            "Parse a CSV file or raw CSV string. Returns column names, "
            "row count, and a preview of the first rows as JSON. "
            "Provide a file path or CSV text."
        ),
    )

"""File Read/Write Tool — let agents read and write local text files.

Writes are restricted to a configurable base directory to prevent
arbitrary filesystem access.
"""

from __future__ import annotations

from pathlib import Path

from agentos.core.tool import Tool


def file_read_tool(*, max_chars: int = 5000) -> Tool:
    """Create a tool that reads a text file and returns its content."""

    def read_file(path: str) -> str:
        """Read a text file and return its contents. Provide the file path."""
        try:
            p = Path(path).expanduser().resolve()
            if not p.is_file():
                return f"Error: File not found: {path}"
            text = p.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_chars:
                return text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"
            return text
        except OSError as e:
            return f"Error reading file: {e}"

    return Tool(
        fn=read_file,
        name="read_file",
        description=(
            "Read a local text file and return its contents. "
            "Provide the file path (absolute or relative)."
        ),
    )


def file_write_tool(*, base_dir: str | None = None) -> Tool:
    """Create a tool that writes text to a file.

    Args:
        base_dir: If provided, all writes are restricted to this directory.
    """

    def write_file(path: str, content: str) -> str:
        """Write text content to a file. Provide the file path and content."""
        try:
            p = Path(path).expanduser().resolve()

            if base_dir:
                allowed = Path(base_dir).expanduser().resolve()
                if not str(p).startswith(str(allowed)):
                    return f"Error: Writes restricted to {base_dir}"

            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} chars to {p}"
        except OSError as e:
            return f"Error writing file: {e}"

    return Tool(
        fn=write_file,
        name="write_file",
        description=(
            "Write text content to a local file. Provide the file path "
            "and the content to write. Creates parent directories if needed."
        ),
    )

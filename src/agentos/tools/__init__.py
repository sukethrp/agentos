"""AgentOS Built-in Tools.

Usage:
    from agentos.tools import get_builtin_tools

    tools = get_builtin_tools()              # dict of all tools
    agent = Agent(tools=list(tools.values()))

    # Or pick specific ones:
    from agentos.tools import calculator_tool, weather_tool, web_search_tool
    agent = Agent(tools=[calculator_tool(), weather_tool()])
"""

from __future__ import annotations
from agentos.core.tool import Tool
from agentos.tools.http_tool import (
    calculator_tool,
    weather_tool,
    web_search_tool,
    news_tool,
    create_api_tool,
)
from agentos.tools.vision_tool import vision_tool
from agentos.tools.document_tool import document_reader_tool, document_qa_tool


def get_builtin_tools() -> dict[str, Tool]:
    """Return a dict of all built-in tools, keyed by name.

    Used by web/app.py and cli.py to avoid redefining tools inline.
    """
    return {
        "calculator": calculator_tool(),
        "weather": weather_tool(),
        "web_search": web_search_tool(),
        "news_search": news_tool(),
        "analyze_image": vision_tool(),
        "read_document": document_reader_tool(),
        "analyze_document": document_qa_tool(),
    }


__all__ = [
    "get_builtin_tools",
    "calculator_tool",
    "weather_tool",
    "web_search_tool",
    "news_tool",
    "create_api_tool",
    "vision_tool",
    "document_reader_tool",
    "document_qa_tool",
]

"""AgentOS Embed — white-label chat widget and Python SDK."""

from agentos.embed.sdk import AgentOSClient
from agentos.embed.widget import generate_snippet, generate_widget, generate_widget_js

__all__ = [
    "AgentOSClient",
    "generate_snippet",
    "generate_widget",
    "generate_widget_js",
]

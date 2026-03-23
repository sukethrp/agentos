"""Demo LLM provider — returns realistic responses without any API keys.

Activated automatically when ``AGENTOS_DEMO_MODE=true``.  The provider
inspects available tools and the user's message to decide whether to
generate a tool-call response or a direct text reply, mimicking the
two-phase pattern of real LLM providers.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Generator

from agentos.core.tool import Tool
from agentos.core.types import AgentEvent, Message, Role, ToolCall

# ---------------------------------------------------------------------------
# Canned tool-call results and text responses
# ---------------------------------------------------------------------------

_TOOL_TRIGGERS: dict[str, list[str]] = {
    "calculator": ["calculate", "math", "compute", "tip", "percent", "%", "+", "*", "sum", "total"],
    "calc": ["calculate", "math", "compute", "tip", "percent", "%", "+", "*", "sum", "total"],
    "get_weather": ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot"],
    "weather": ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot"],
    "web_search": ["search", "look up", "find", "google", "latest", "news"],
    "company_lookup": ["company", "about", "tell me about", "info on"],
}

_DEMO_TOOL_ARGS: dict[str, dict] = {
    "calculator": {"expression": "85.50 * 0.15"},
    "calc": {"expression": "85.50 * 0.15"},
    "get_weather": {"city": "San Francisco"},
    "weather": {"city": "San Francisco"},
    "web_search": {"query": "latest AI agent frameworks 2026"},
    "company_lookup": {"company_name": "Anthropic"},
}

_FALLBACK_RESPONSES = [
    "Based on my analysis, here's what I found:\n\n"
    "AgentOS provides a comprehensive platform for building, testing, and "
    "deploying AI agents. It includes governance controls, monitoring, and "
    "a marketplace for sharing agent templates.\n\n"
    "Key features include:\n"
    "- **Simulation Sandbox** for testing agents against scenarios\n"
    "- **Live Dashboard** for real-time monitoring\n"
    "- **Governance Engine** with budget limits and permissions\n"
    "- **Multi-provider support** for OpenAI, Anthropic, and Ollama",

    "I'd be happy to help with that! Here's a summary:\n\n"
    "The task has been completed successfully. AgentOS makes it easy to "
    "build production-ready AI agents with just a few lines of code. "
    "The platform handles tool orchestration, cost tracking, and safety "
    "controls automatically.\n\n"
    "Would you like me to go into more detail on any specific aspect?",

    "Great question! Let me break this down:\n\n"
    "1. **Agent Definition** — Define tools and system prompts\n"
    "2. **Testing** — Run sandbox scenarios before deploying\n"
    "3. **Deployment** — Push to production with governance controls\n"
    "4. **Monitoring** — Track costs, latency, and quality in real-time\n\n"
    "This workflow ensures agents are safe and reliable before they "
    "interact with real users.",
]


def _pick_tool_call(
    user_msg: str, tools: list[Tool]
) -> ToolCall | None:
    """Decide whether to call a tool based on the user message."""
    msg_lower = user_msg.lower()
    for tool in tools:
        triggers = _TOOL_TRIGGERS.get(tool.name, [])
        if any(t in msg_lower for t in triggers):
            args = _DEMO_TOOL_ARGS.get(tool.name, {})
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=tool.name,
                arguments=args,
            )
    return None


def _has_tool_results(messages: list[dict]) -> bool:
    """Check if the conversation already contains tool results."""
    return any(m.get("role") == "tool" for m in messages)


def _synthesize_response(messages: list[dict]) -> str:
    """Build a final response, incorporating tool results if present."""
    tool_results = [m for m in messages if m.get("role") == "tool"]
    if tool_results:
        parts = []
        for tr in tool_results:
            content = tr.get("content", "")
            parts.append(content)
        result_text = ", ".join(parts)
        return (
            f"Here's what I found using the tools:\n\n"
            f"**Result:** {result_text}\n\n"
            f"Let me know if you'd like me to look into anything else!"
        )
    return random.choice(_FALLBACK_RESPONSES)


def _make_event(
    agent_name: str,
    event_type: str,
    latency_ms: float,
    tokens: int = 0,
) -> AgentEvent:
    return AgentEvent(
        agent_name=agent_name,
        event_type=event_type,
        tokens_used=tokens,
        cost_usd=round(tokens * 0.000002, 6),
        latency_ms=round(latency_ms, 2),
        data={"provider": "demo", "model": "demo-mode"},
    )


# ---------------------------------------------------------------------------
# Public API (matches openai_provider / anthropic_provider signatures)
# ---------------------------------------------------------------------------


def call_demo(
    messages: list[dict],
    tools: list[Tool],
    model: str = "demo",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    """Return a mock LLM response, optionally with tool calls."""
    start = time.time()

    user_msgs = [m for m in messages if m.get("role") == "user"]
    last_user_msg = user_msgs[-1]["content"] if user_msgs else ""

    if tools and not _has_tool_results(messages):
        tc = _pick_tool_call(last_user_msg, tools)
        if tc:
            latency = (time.time() - start) * 1000 + random.uniform(80, 200)
            msg = Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[tc],
            )
            event = _make_event(agent_name, "llm_call", latency, tokens=150)
            return msg, event

    content = _synthesize_response(messages)
    latency = (time.time() - start) * 1000 + random.uniform(100, 300)
    tokens = len(content.split()) * 2
    msg = Message(role=Role.ASSISTANT, content=content)
    event = _make_event(agent_name, "llm_call", latency, tokens=tokens)
    return msg, event


def call_demo_stream(
    messages: list[dict],
    tools: list[Tool],
    model: str = "demo",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> Generator[str | tuple[str, Message, AgentEvent], None, None]:
    """Streaming version — yields tokens then a final tuple."""
    start = time.time()

    user_msgs = [m for m in messages if m.get("role") == "user"]
    last_user_msg = user_msgs[-1]["content"] if user_msgs else ""

    if tools and not _has_tool_results(messages):
        tc = _pick_tool_call(last_user_msg, tools)
        if tc:
            latency = (time.time() - start) * 1000 + random.uniform(80, 200)
            msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[tc])
            event = _make_event(agent_name, "llm_call", latency, tokens=150)
            yield ("tool_calls", msg, event)
            return

    content = _synthesize_response(messages)
    words = content.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")

    latency = (time.time() - start) * 1000
    tokens = len(words) * 2
    msg = Message(role=Role.ASSISTANT, content=content)
    event = _make_event(agent_name, "llm_call", latency, tokens=tokens)
    yield ("done", msg, event)

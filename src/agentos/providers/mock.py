"""Mock provider for zero-config demos and testing.

A fully-featured mock LLM provider that implements the same
:class:`~agentos.providers.base.BaseProvider` interface as the real providers
and returns realistic fake responses **without any API keys**.  Useful for:

- **Local development** without burning API credits
- **CI/CD pipelines** where secrets aren't available
- **Demos and workshops** for first-time users
- **Integration tests** that need deterministic-ish outputs

Activate via environment variable::

    AGENTOS_DEMO_MODE=true python -m agentos

Or use ``MockProvider`` directly::

    from agentos.providers.mock import MockProvider
    provider = MockProvider()
    msg, event = provider.chat_completion(messages, tools)

**Response simulation logic:**

The mock follows a two-phase strategy that mirrors what a real LLM does:

1. **Tool-call phase** — if tools are available and the conversation does
   not yet contain any ``tool`` messages, the provider scans the last user
   message for keyword triggers (e.g. "calculate", "weather").  When a
   trigger matches a registered tool, a synthetic :class:`ToolCall` is
   returned so the agent's ReAct loop will execute the tool and feed the
   result back.

2. **Response phase** — once tool results are present (or no tool matched),
   a template-based response is generated.  Templates are keyed by detected
   topic (``"calculator"``, ``"weather"``, ``"general"``) and randomly
   chosen to add variety.  ``{tool_result}`` placeholders are interpolated
   with actual tool output.

Artificial latency (200-500 ms sync, 10-30 ms per token streaming) is
injected so the UI behaves realistically.  Token counts are estimated at
~1 token per 4 characters, and cost is fixed at ``$0.0001`` per response.
"""

from __future__ import annotations

import random
import re
import time
import uuid
from typing import AsyncGenerator, Generator

from agentos.core.tool import Tool
from agentos.core.types import AgentEvent, Message, Role, ToolCall
from agentos.providers.base import BaseProvider
from agentos.tools.safe_math import safe_eval_math

# ---------------------------------------------------------------------------
# Response templates keyed by topic detected in the user message
# ---------------------------------------------------------------------------

_TOPIC_RESPONSES: dict[str, list[str]] = {
    "calculator": [
        "Based on my calculation, the result is **{tool_result}**.\n\n"
        "I used the calculator tool to compute this. Let me know if you'd "
        "like me to run any other calculations!",
        "The answer is **{tool_result}**. I computed this using the "
        "calculator tool. Would you like to try a different expression?",
    ],
    "weather": [
        "Here's the current weather:\n\n**{tool_result}**\n\n"
        "Would you like a more detailed forecast or weather for another city?",
        "According to the weather data: **{tool_result}**\n\n"
        "Let me know if you need forecasts for additional locations!",
    ],
    "general": [
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
    ],
}

# Maps tool names to keywords that trigger them
_TOOL_TRIGGERS: dict[str, list[str]] = {
    "calculator": ["calculate", "math", "compute", "tip", "percent", "%", "+", "*", "sum", "total"],
    "calc": ["calculate", "math", "compute", "tip", "percent", "%", "+", "*", "sum", "total"],
    "get_weather": ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot"],
    "weather": ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot"],
    "web_search": ["search", "look up", "find", "google", "latest", "news"],
    "company_lookup": ["company", "about", "tell me about", "info on"],
}

# Default arguments for demo tool calls (calculator uses live expression parsing)
_DEMO_TOOL_ARGS: dict[str, dict] = {
    "get_weather": {"city": "San Francisco"},
    "weather": {"city": "San Francisco"},
    "web_search": {"query": "latest AI agent frameworks 2026"},
    "company_lookup": {"company_name": "Anthropic"},
}

MOCK_COST_PER_RESPONSE = 0.0001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Roughly 1 token per 4 characters, matching real tokeniser heuristics."""
    return max(1, len(text) // 4)


_PERCENT_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)


def _extract_expression(user_msg: str) -> str | None:
    """Pull a real arithmetic expression out of the user's message.

    Handles "X% of Y" phrasing and any inline arithmetic; returns None when
    nothing evaluable is found so the caller can fall back to a demo default.
    """
    pct = _PERCENT_OF.search(user_msg)
    if pct:
        return f"{pct.group(2)} * {pct.group(1)} / 100"
    for candidate in sorted(re.findall(r"[-+*/().\d\s]+", user_msg), key=len, reverse=True):
        candidate = candidate.strip()
        if any(op in candidate for op in "+-*/") and any(c.isdigit() for c in candidate):
            try:
                safe_eval_math(candidate)
                return candidate
            except ValueError:
                continue
    return None


def _pick_tool_call(user_msg: str, tools: list[Tool]) -> ToolCall | None:
    """Decide whether to simulate a tool call based on message keywords.

    Iterates over available tools and checks the user message for keyword
    triggers defined in ``_TOOL_TRIGGERS``.  The first matching tool wins —
    we don't attempt multi-tool calls because the mock is meant to exercise
    the single-tool-call code path, which covers most demo scenarios.

    Args:
        user_msg: The latest user message text.
        tools: Tools currently registered on the agent.

    Returns:
        A synthetic :class:`ToolCall` if a trigger matched, else ``None``.
    """
    msg_lower = user_msg.lower()
    for tool in tools:
        triggers = _TOOL_TRIGGERS.get(tool.name, [])
        if any(t in msg_lower for t in triggers):
            args = dict(_DEMO_TOOL_ARGS.get(tool.name, {}))
            if tool.name in ("calculator", "calc"):
                expr = _extract_expression(user_msg)
                if expr:
                    args["expression"] = expr
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=tool.name,
                arguments=args,
            )
    return None


def _has_tool_results(messages: list[dict]) -> bool:
    """Check if the conversation already contains tool results."""
    return any(m.get("role") == "tool" for m in messages)


def _detect_topic(messages: list[dict]) -> str:
    """Detect the response topic from the conversation.

    Walks the messages backward to find the most recent assistant tool call
    and maps the tool name to a topic key (``"calculator"``, ``"weather"``,
    or ``"general"``).  This determines which response template to use.
    """
    tool_results = [m for m in messages if m.get("role") == "tool"]
    if tool_results:
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    # Handle both dict and object formats
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "") if isinstance(fn, dict) else tc.get("name", "")
                    else:
                        name = getattr(tc, "name", "")
                    if name in ("calculator", "calc"):
                        return "calculator"
                    if name in ("get_weather", "weather"):
                        return "weather"
    return "general"


def _synthesize_response(messages: list[dict]) -> str:
    """Build a final text response, interpolating tool results into templates.

    Picks a random template from ``_TOPIC_RESPONSES`` for the detected topic
    and fills in ``{tool_result}`` placeholders with any tool-role messages
    in the conversation.
    """
    topic = _detect_topic(messages)
    templates = _TOPIC_RESPONSES.get(topic, _TOPIC_RESPONSES["general"])
    template = random.choice(templates)

    tool_results = [m for m in messages if m.get("role") == "tool"]
    if tool_results and "{tool_result}" in template:
        result_text = ", ".join(m.get("content", "") for m in tool_results)
        return template.format(tool_result=result_text)
    if "{tool_result}" in template:
        return template.format(tool_result="(no data)")
    return template


def _make_event(
    agent_name: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    has_tool_calls: bool = False,
) -> AgentEvent:
    """Build an AgentEvent with realistic-looking metrics."""
    total_tokens = prompt_tokens + completion_tokens
    cost = MOCK_COST_PER_RESPONSE
    return AgentEvent(
        agent_name=agent_name,
        event_type="llm_call",
        tokens_used=total_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
        data={
            "provider": "mock",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "has_tool_calls": has_tool_calls,
        },
    )


# ---------------------------------------------------------------------------
# MockProvider class (implements BaseProvider interface)
# ---------------------------------------------------------------------------


class MockProvider(BaseProvider):
    """A mock LLM provider that returns realistic responses without API keys.

    Implements the same ``BaseProvider`` interface as the real providers so it
    can be used as a drop-in replacement anywhere AgentOS expects a provider.

    Behaviour:
    - Inspects the user message for keywords ("calculate", "weather", etc.)
      and simulates the appropriate tool call if matching tools are available.
    - After tool results are fed back, generates a natural-language summary.
    - For general queries, returns a helpful paragraph about AgentOS.
    - Adds a 200-500 ms delay to mimic real LLM latency.
    - Reports ~1 token per 4 characters and $0.0001 per response.
    """

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[Tool],
        model: str = "mock",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        agent_name: str = "agent",
    ) -> tuple[Message, AgentEvent]:
        """Return a mock chat completion, optionally with tool calls.

        Delegates to the module-level :func:`call_mock` so both the class
        and function APIs share the same simulation logic.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Available tools for keyword-trigger matching.
            model: Model identifier (ignored).
            temperature: Ignored.
            max_tokens: Ignored.
            agent_name: Attribution label for the ``AgentEvent``.

        Returns:
            A ``(Message, AgentEvent)`` tuple.
        """
        msg, event = call_mock(
            messages, tools,
            model=model, temperature=temperature,
            max_tokens=max_tokens, agent_name=agent_name,
        )
        return msg, event

    async def stream(
        self,
        messages: list[dict],
        tools: list[Tool],
        model: str = "mock",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        agent_name: str = "agent",
    ) -> AsyncGenerator[str | tuple[str, Message, AgentEvent], None]:
        """Yield tokens with small delays, then a final ``(tag, Message, Event)``.

        Wraps the synchronous :func:`call_mock_stream` generator so it can
        be consumed via ``async for`` in the streaming code path.
        """
        for item in call_mock_stream(
            messages, tools,
            model=model, temperature=temperature,
            max_tokens=max_tokens, agent_name=agent_name,
        ):
            yield item


# ---------------------------------------------------------------------------
# Module-level functions (match openai_provider / demo_provider signatures)
# ---------------------------------------------------------------------------


def call_mock(
    messages: list[dict],
    tools: list[Tool],
    model: str = "mock",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> tuple[Message, AgentEvent]:
    """Synchronous mock LLM call — returns ``(Message, AgentEvent)``.

    Implements the two-phase strategy (tool call → text response) described
    in the module docstring.  A 200-500 ms sleep simulates network latency.

    Args:
        messages: Conversation history in OpenAI message format.
        tools: Available tools; triggers are matched against the last user msg.
        model: Model identifier (ignored; always behaves the same).
        temperature: Ignored — responses are template-based.
        max_tokens: Ignored — response length is fixed by templates.
        agent_name: Used in the returned ``AgentEvent`` for attribution.

    Returns:
        A ``(Message, AgentEvent)`` tuple matching the real provider contract.
    """
    start = time.time()

    user_msgs = [m for m in messages if m.get("role") == "user"]
    last_user_msg = user_msgs[-1]["content"] if user_msgs else ""

    # Simulate realistic latency (200-500ms)
    delay = random.uniform(0.20, 0.50)
    time.sleep(delay)

    prompt_tokens = _estimate_tokens(
        " ".join(m.get("content", "") or "" for m in messages)
    )

    if tools and not _has_tool_results(messages):
        tc = _pick_tool_call(last_user_msg, tools)
        if tc:
            latency = (time.time() - start) * 1000
            msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[tc])
            event = _make_event(
                agent_name, model, latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=20,
                has_tool_calls=True,
            )
            return msg, event

    content = _synthesize_response(messages)
    latency = (time.time() - start) * 1000
    completion_tokens = _estimate_tokens(content)

    msg = Message(role=Role.ASSISTANT, content=content)
    event = _make_event(
        agent_name, model, latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return msg, event


def call_mock_stream(
    messages: list[dict],
    tools: list[Tool],
    model: str = "mock",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent_name: str = "agent",
) -> Generator[str | tuple[str, Message, AgentEvent], None, None]:
    """Streaming mock — yields string tokens then a final (tag, Message, Event).

    Simulates realistic streaming by yielding words with small delays
    (10-30 ms per token) so the UI feels like a real LLM.
    """
    start = time.time()

    user_msgs = [m for m in messages if m.get("role") == "user"]
    last_user_msg = user_msgs[-1]["content"] if user_msgs else ""

    prompt_tokens = _estimate_tokens(
        " ".join(m.get("content", "") or "" for m in messages)
    )

    if tools and not _has_tool_results(messages):
        tc = _pick_tool_call(last_user_msg, tools)
        if tc:
            time.sleep(random.uniform(0.10, 0.25))
            latency = (time.time() - start) * 1000
            msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[tc])
            event = _make_event(
                agent_name, model, latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=20,
                has_tool_calls=True,
            )
            yield ("tool_calls", msg, event)
            return

    content = _synthesize_response(messages)
    words = content.split(" ")
    for i, word in enumerate(words):
        time.sleep(random.uniform(0.01, 0.03))
        yield word + (" " if i < len(words) - 1 else "")

    latency = (time.time() - start) * 1000
    completion_tokens = _estimate_tokens(content)

    msg = Message(role=Role.ASSISTANT, content=content)
    event = _make_event(
        agent_name, model, latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    yield ("done", msg, event)

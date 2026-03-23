"""ReAct-style tool-calling loop with provider abstraction.

The Agent class is the central execution engine of AgentOS.  It orchestrates
a **ReAct loop** (Reason → Act → Observe → Repeat):

1. **Query** — the user's natural-language message is combined with the
   system prompt and conversation history.
2. **LLM call** — the message list and tool JSON-Schema definitions are sent
   to the configured provider (OpenAI, Anthropic, Ollama, or Mock).
3. **Tool dispatch** — if the LLM response contains ``tool_calls``, each
   tool is executed (concurrently where possible) and the results appended
   to the conversation as ``tool`` messages.
4. **Repeat** — steps 2-3 repeat until the LLM produces a final text
   response with no tool calls, or ``max_iterations`` is reached.

**Why ``max_iterations``?**  Without a cap the loop could spin forever if the
LLM keeps requesting tools without converging (e.g. hallucinated tool names,
circular reasoning).  The default of 10 balances complex multi-step tasks
against runaway cost.

**Cost tracking:**  Every LLM call returns token counts and a USD cost
estimate in its ``AgentEvent``.  These events are accumulated in
``self.events`` so callers (the CLI, web dashboard, and ``GovernedAgent``)
can aggregate total spend.  The per-token price is looked up from a static
pricing table inside each provider, *not* from a live billing API, so the
figures are estimates — but accurate enough for budget enforcement.

Design decisions:
- **Provider-agnostic** — works with any class implementing ``BaseProvider``.
- **Streaming-first** — ``run()`` supports both sync and streaming modes.
- **Observable** — every action emits an ``AgentEvent`` for monitoring.
- **Cacheable** — deterministic tool calls are memoised via a TTL cache to
  avoid redundant executions within a session.

.. todo::
   TODO(#42): Integrate MCP transport so agents can expose / consume tools
   over the Model Context Protocol, enabling cross-process tool sharing
   with Claude Desktop, Cursor, and other MCP-aware clients.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Generator
from cachetools import TTLCache

if TYPE_CHECKING:
    from agentos.mcp import MCPServer
from agentos.core.types import (
    AgentConfig,
    AgentEvent,
    Message,
    Role,
    ToolCall,
    ToolExecutionContext,
    ToolResult,
)
from agentos.core.tool import Tool
from agentos.core.memory import Memory
from agentos.providers.router import call_model as call_llm
from agentos.providers.router import call_model_stream as call_llm_stream
from agentos.logging import correlation, get_logger
from agentos.telemetry import (
    agent_span,
    llm_span,
    record_llm_metrics,
    record_tool_metrics,
    tool_span,
)

_log = get_logger("agentos.agent")

# Short-lived cache for deterministic tool calls (e.g. calculator, lookups).
# TTL of 300s avoids stale results while saving redundant executions when the
# LLM re-invokes the same tool with identical arguments within a single session.
_tool_cache: TTLCache[str, str] = TTLCache(maxsize=256, ttl=300)


def _cache_key(tool_name: str, args: dict) -> str:
    """Produce a stable hash for a (tool, args) pair to use as a cache key.

    Arguments are JSON-serialised with sorted keys so that ``{"a":1, "b":2}``
    and ``{"b":2, "a":1}`` yield the same digest.
    """
    return hashlib.sha256(
        (tool_name + json.dumps(args, sort_keys=True)).encode()
    ).hexdigest()


class Agent:
    """A provider-agnostic, tool-calling agent built on the ReAct loop.

    The agent sends the conversation history and tool schemas to an LLM,
    executes any requested tool calls, feeds the results back, and repeats
    until the LLM produces a final text answer or the iteration cap is hit.

    Example::

        from agentos.core.agent import Agent
        from agentos.core.tool import Tool

        agent = Agent(
            name="weather-bot",
            model="gpt-4o-mini",
            tools=[Tool(name="get_weather", fn=get_weather, description="...")],
        )
        response = agent.run("What's the weather in NYC?")
        print(response.content)
    """

    # TODO(#11): Add agent-to-agent message passing here

    def __init__(
        self,
        name: str = "agent",
        model: str = "gpt-4o-mini",
        tools: list[Tool] | None = None,
        system_prompt: str = "You are a helpful assistant. Use tools when needed.",
        max_iterations: int = 10,
        temperature: float = 0.7,
        memory: Memory | None = None,
    ) -> None:
        """Initialise a new Agent instance.

        Args:
            name: Human-readable identifier, also used in logs and telemetry.
            model: LLM model identifier (e.g. ``"gpt-4o-mini"``, ``"claude-3"``).
                   Resolved at call-time by the provider router.
            tools: Tools the agent may invoke.  Pass ``None`` for a chat-only agent.
            system_prompt: Injected as the first system message on every run.
            max_iterations: Safety cap on the ReAct loop to prevent infinite
                            tool-calling cycles.
            temperature: LLM sampling temperature (0 = deterministic).
            memory: Conversation memory store.  A fresh ``Memory()`` is created
                    if none is supplied.
        """
        self.config = AgentConfig(
            name=name,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            temperature=temperature,
        )
        self.tools = tools or []
        self._tool_map = {t.name: t for t in self.tools}
        self.events: list[AgentEvent] = []
        self.messages: list[dict] = []
        self.memory = memory or Memory()
        self._session_id = str(uuid.uuid4())

    def run(
        self, user_input: str, stream: bool = False
    ) -> Message | Generator[str | Message, None, None]:
        """Execute the agent's ReAct loop for a single user turn.

        Args:
            user_input: The user's natural-language query or instruction.
            stream: When ``True`` the method returns a generator that yields
                    string tokens as they arrive from the LLM, with the final
                    ``Message`` yielded last.

        Returns:
            A ``Message`` (sync) or a ``Generator[str | Message]`` (streaming)
            containing the agent's final response.

        Example::

            # Synchronous
            msg = agent.run("Summarise this PDF")

            # Streaming
            for chunk in agent.run("Summarise this PDF", stream=True):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
        """
        # Build the message list from memory so the LLM has access to prior
        # conversation context and extracted facts.
        self.messages = self.memory.build_messages(
            self.config.system_prompt,
            user_input,
        )
        self.events = []

        if not stream:
            return self._run_sync(user_input)
        return self._run_stream(user_input)

    def _run_sync(self, user_input: str) -> Message:
        """Execute the ReAct loop synchronously, returning the final message.

        Iterates up to ``max_iterations`` times, calling the LLM and executing
        any requested tools on each pass.  Returns as soon as the LLM produces
        a response with no tool calls (i.e. a final answer).

        Args:
            user_input: The raw user query (used for logging and memory).

        Returns:
            The LLM's final ``Message`` once it stops requesting tools.
        """
        print(f"\n🤖 [{self.config.name}] Processing: {user_input}")
        print("-" * 60)

        with (
            agent_span(self.config.name, self.config.model, user_input) as _span,
            correlation(agent=self.config.name, model=self.config.model, session=self._session_id) as ctx,
        ):
            _log.info("agent.run.start", extra=ctx(input=user_input[:200]))

            # Cap at max_iterations to prevent infinite loops if the LLM
            # keeps calling tools without converging on a final answer.
            for i in range(self.config.max_iterations):
                # TODO(#7): Emit OpenTelemetry spans for each tool call
                _log.debug("llm.call.start", extra=ctx(step=i + 1))
                with llm_span(self.config.model):
                    msg, event = call_llm(
                        model=self.config.model,
                        messages=self.messages,
                        tools=self.tools,
                        temperature=self.config.temperature,
                        agent_name=self.config.name,
                    )
                self.events.append(event)
                # Token costs vary by model — we look up the per-token rate
                # from the provider's pricing table inside record_llm_metrics.
                record_llm_metrics(event)
                _log.info(
                    "llm.call.end",
                    extra=ctx(
                        step=i + 1,
                        tokens=event.tokens_used,
                        cost=event.cost_usd,
                        latency_ms=event.latency_ms,
                        provider=event.data.get("provider", ""),
                    ),
                )

                if not msg.tool_calls:
                    print(
                        f"\n✅ Final Answer (${event.cost_usd:.4f}, {event.latency_ms:.0f}ms):"
                    )
                    print(msg.content)
                    self._print_summary()
                    if _span is not None:
                        _span.set_attribute("agentos.agent.output", (msg.content or "")[:500])

                    self.memory.add_exchange(user_input, msg.content or "")
                    self.memory.extract_facts_from_response(user_input, msg.content or "")

                    _log.info(
                        "agent.run.end",
                        extra=ctx(
                            output=(msg.content or "")[:200],
                            total_tokens=sum(e.tokens_used for e in self.events),
                            total_cost=sum(e.cost_usd for e in self.events),
                        ),
                    )
                    return msg

                # LLM wants to call tool(s) — append its response to the
                # conversation so the next LLM call sees its own reasoning.
                print(f"\n🔧 Step {i + 1}: Using {len(msg.tool_calls)} tool(s)")
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": str(tc.arguments),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                budget_remaining = 0.0
                exec_ctx = ToolExecutionContext(
                    agent_id=self.config.name,
                    session_id=self._session_id,
                    budget_remaining=budget_remaining,
                )
                results = self._execute_tools_batch(msg.tool_calls, exec_ctx)
                for tc, (result_str, latency_ms) in zip(msg.tool_calls, results):
                    print(f"   🔨 {tc.name}({tc.arguments}) → {result_str[:80]}")
                    record_tool_metrics(self.config.name, tc.name, latency_ms)
                    _log.info(
                        "tool.execute",
                        extra=ctx(
                            tool=tc.name,
                            latency_ms=latency_ms,
                            error=result_str[:120] if result_str.startswith("ERROR") else None,
                        ),
                    )
                    self.events.append(
                        AgentEvent(
                            agent_name=self.config.name,
                            event_type="tool_call",
                            data={
                                "tool": tc.name,
                                "args": tc.arguments,
                                "result": result_str[:200],
                            },
                            latency_ms=latency_ms,
                        )
                    )
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                    )

            _log.warning("agent.run.max_iterations", extra=ctx())
            print("⚠️ Max iterations reached")
            return Message(role=Role.ASSISTANT, content="Could not complete the task.")

    def _run_stream(self, user_input: str) -> Generator[str | Message, None, None]:
        """Streaming variant of the ReAct loop.

        Yields string tokens as they arrive from the LLM so callers can
        display incremental output.  The final ``Message`` object is yielded
        last once the LLM produces a complete, tool-call-free response.

        Args:
            user_input: The raw user query.

        Yields:
            ``str`` chunks during generation, then a single ``Message`` at the end.
        """
        for i in range(self.config.max_iterations):
            stream = call_llm_stream(
                model=self.config.model,
                messages=self.messages,
                tools=self.tools,
                temperature=self.config.temperature,
                agent_name=self.config.name,
            )
            last_item = None
            for chunk in stream:
                if isinstance(chunk, str):
                    yield chunk
                else:
                    last_item = chunk

            if last_item is None:
                return Message(role=Role.ASSISTANT, content="")

            tag, msg, event = last_item
            self.events.append(event)

            if tag == "done":
                self.memory.add_exchange(user_input, msg.content or "")
                self.memory.extract_facts_from_response(user_input, msg.content or "")
                yield msg
                return

            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": str(tc.arguments),
                            },
                        }
                        for tc in (msg.tool_calls or [])
                    ],
                }
            )
            ctx = ToolExecutionContext(
                agent_id=self.config.name,
                session_id=self._session_id,
                budget_remaining=0.0,
            )
            results = self._execute_tools_batch(msg.tool_calls or [], ctx)
            for tc, (result_str, latency_ms) in zip(msg.tool_calls or [], results):
                self.events.append(
                    AgentEvent(
                        agent_name=self.config.name,
                        event_type="tool_call",
                        data={
                            "tool": tc.name,
                            "args": tc.arguments,
                            "result": result_str[:200],
                        },
                        latency_ms=latency_ms,
                    )
                )
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                )

        yield Message(role=Role.ASSISTANT, content="Could not complete the task.")

    def _execute_tools_batch(
        self,
        tool_calls: list[ToolCall],
        ctx: ToolExecutionContext,
    ) -> list[tuple[str, float]]:
        """Run all tool calls concurrently via an async event loop.

        This is the sync entry-point; it spins up ``asyncio.run`` internally
        so callers don't need to be async-aware.

        Args:
            tool_calls: Tool invocations requested by the LLM.
            ctx: Execution context carrying agent/session IDs and budget state.

        Returns:
            A list of ``(result_string, latency_ms)`` tuples, one per tool call,
            in the same order as ``tool_calls``.
        """
        # TODO: Support parallel tool execution when tools are independent
        return asyncio.run(self._execute_tools_async(tool_calls, ctx))

    async def _execute_tools_async(
        self,
        tool_calls: list[ToolCall],
        ctx: ToolExecutionContext,
    ) -> list[tuple[str, float]]:
        """Concurrently execute tool calls with caching, retries, and timeouts.

        Each call is dispatched as its own task via ``asyncio.gather`` so
        independent tools run in parallel.  Results are cached by a
        content-hash of (tool_name, args) to avoid redundant executions.

        Args:
            tool_calls: Tool invocations requested by the LLM.
            ctx: Execution context for agent/session metadata.

        Returns:
            Ordered list of ``(result_string, latency_ms)`` tuples.
        """
        from agentos.monitor.ws_manager import broadcast_tool_event

        async def run_one(tc: ToolCall) -> tuple[str, float]:
            tool = self._tool_map.get(tc.name)
            if not tool:
                return (f"ERROR: Tool '{tc.name}' not found", 0.0)
            ck = _cache_key(tc.name, tc.arguments)
            # Return cached results for deterministic tools to save latency
            # and avoid duplicate side-effects (e.g. duplicate web requests).
            if ck in _tool_cache:
                cached = _tool_cache[ck]
                await broadcast_tool_event(ctx.agent_id, tc.name, cached, 0.0)
                return (cached, 0.0)
            last_err = None
            with tool_span(tc.name):
                for attempt in range(tool.max_retries + 1):
                    try:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(self._execute_tool_with_retry, tool, tc),
                            timeout=tool.timeout_seconds,
                        )
                        result_str = result.result
                        latency_ms = result.latency_ms
                        _tool_cache[ck] = result_str
                        await broadcast_tool_event(
                            ctx.agent_id, tc.name, result_str, latency_ms
                        )
                        return (result_str, latency_ms)
                    # Catch tool execution errors gracefully so one bad tool
                    # doesn't crash the entire agent run.
                    except asyncio.TimeoutError as e:
                        last_err = e
                        # Exponential back-off before retrying.
                        if attempt < tool.max_retries:
                            await asyncio.sleep(2**attempt)
                    except Exception as e:
                        last_err = e
                        if attempt < tool.max_retries:
                            await asyncio.sleep(2**attempt)
            err_str = str(last_err) if last_err else "Unknown error"
            return (f"ERROR: {err_str}", 0.0)

        return list(await asyncio.gather(*[run_one(tc) for tc in tool_calls]))

    def _execute_tool_with_retry(self, tool: Tool, tc: ToolCall) -> ToolResult:
        """Invoke a single tool function and wrap the outcome in a ``ToolResult``.

        This runs on a worker thread (via ``asyncio.to_thread``) so blocking
        tool functions don't stall the async event loop.  Exceptions are
        caught and returned as ``ERROR:`` strings rather than propagated,
        keeping the ReAct loop alive even when a tool misbehaves.

        Args:
            tool: The ``Tool`` instance containing the callable.
            tc: The ``ToolCall`` with the function name and arguments.

        Returns:
            A ``ToolResult`` with either the stringified return value or an
            error description, plus wall-clock latency in milliseconds.
        """
        import time

        start = time.time()
        try:
            result = tool.fn(**tc.arguments)
            latency = (time.time() - start) * 1000
            return ToolResult(
                tool_call_id=tc.id,
                name=tool.name,
                result=str(result),
                latency_ms=round(latency, 2),
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return ToolResult(
                tool_call_id=tc.id,
                name=tool.name,
                result=f"ERROR: {e}",
                latency_ms=round(latency, 2),
            )

    def delegate(self, to_agent: Agent, task: str) -> str:
        """Delegate a subtask to another agent and return its text response.

        This is a simple point-to-point delegation.  For multi-agent
        orchestration with cost tracking and shared context, use
        :class:`~agentos.mesh.AgentMesh` instead.

        Args:
            to_agent: The ``Agent`` instance that will handle the subtask.
            task: Natural-language description of the delegated work.

        Returns:
            The delegate agent's final text response.

        Example::

            researcher = Agent(name="researcher", model="gpt-4o-mini")
            answer = main_agent.delegate(researcher, "Find the GDP of France")
        """
        response = to_agent.run(task)
        return response.content or ""

    def as_mcp_server(self, name: str | None = None) -> MCPServer:
        """Return an MCPServer exposing this agent's tools over MCP.

        Wraps the agent's tool list into a Model Context Protocol server so
        external clients can discover and invoke tools over a standard
        transport (stdio / SSE).

        Args:
            name: Optional server name.  Defaults to the agent's ``config.name``.

        Returns:
            A configured ``MCPServer`` instance ready to be started.

        Note:
            Requires the ``mcp`` extra: ``pip install 'agentos-platform[mcp]'``
        """
        from agentos.mcp import MCPServer

        return MCPServer.from_agent(self, name=name)

    def _print_summary(self) -> None:
        """Print a human-readable summary of the completed agent run.

        Aggregates cost, token, and latency data across all recorded events
        (both LLM calls and tool calls) and prints them to stdout.  This is
        intended for interactive / CLI usage, not for programmatic consumption
        — use ``self.events`` directly for that.
        """
        total_cost = sum(e.cost_usd for e in self.events)
        total_tokens = sum(e.tokens_used for e in self.events)
        total_latency = sum(e.latency_ms for e in self.events)
        tool_calls = sum(1 for e in self.events if e.event_type == "tool_call")
        llm_calls = sum(1 for e in self.events if e.event_type == "llm_call")

        print(f"\n{'=' * 60}")
        print("📊 Agent Run Summary")
        print(f"{'=' * 60}")
        print(f"   LLM calls:    {llm_calls}")
        print(f"   Tool calls:   {tool_calls}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Total cost:   ${total_cost:.4f}")
        print(f"   Total time:   {total_latency:.0f}ms")
        print(f"{'=' * 60}")

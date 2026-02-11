from __future__ import annotations
from typing import Generator
from agentos.core.types import AgentConfig, AgentEvent, Message, Role
from agentos.core.tool import Tool
from agentos.core.memory import Memory
from agentos.providers.router import call_model as call_llm
from agentos.providers.router import call_model_stream as call_llm_stream


class Agent:
    def __init__(
        self,
        name: str = "agent",
        model: str = "gpt-4o-mini",
        tools: list[Tool] | None = None,
        system_prompt: str = "You are a helpful assistant. Use tools when needed.",
        max_iterations: int = 10,
        temperature: float = 0.7,
        memory: Memory | None = None,
    ):
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

    def run(self, user_input: str, stream: bool = False) -> Message | Generator[str | Message, None, None]:
        """Run the agent. If stream=True, yields tokens as they arrive, then returns the final Message."""
        # Build messages with memory context
        self.messages = self.memory.build_messages(
            self.config.system_prompt,
            user_input,
        )
        self.events = []

        if not stream:
            return self._run_sync(user_input)
        return self._run_stream(user_input)

    def _run_sync(self, user_input: str) -> Message:
        print(f"\n🤖 [{self.config.name}] Processing: {user_input}")
        print("-" * 60)

        for i in range(self.config.max_iterations):
            msg, event = call_llm(
                model=self.config.model,
                messages=self.messages,
                tools=self.tools,
                temperature=self.config.temperature,
                agent_name=self.config.name,
            )
            self.events.append(event)

            if not msg.tool_calls:
                print(f"\n✅ Final Answer (${event.cost_usd:.4f}, {event.latency_ms:.0f}ms):")
                print(msg.content)
                self._print_summary()

                # Store in memory
                self.memory.add_exchange(user_input, msg.content or "")
                self.memory.extract_facts_from_response(user_input, msg.content or "")

                return msg

            print(f"\n🔧 Step {i+1}: Using {len(msg.tool_calls)} tool(s)")

            self.messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool = self._tool_map.get(tc.name)
                if not tool:
                    result_str = f"ERROR: Tool '{tc.name}' not found"
                    print(f"   ❌ {tc.name}: not found")
                else:
                    result = tool.execute(tc)
                    result_str = result.result
                    print(f"   🔨 {tc.name}({tc.arguments}) → {result_str[:80]}")

                    self.events.append(AgentEvent(
                        agent_name=self.config.name,
                        event_type="tool_call",
                        data={"tool": tc.name, "args": tc.arguments, "result": result_str[:200]},
                        latency_ms=result.latency_ms,
                    ))

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        print("⚠️ Max iterations reached")
        return Message(role=Role.ASSISTANT, content="Could not complete the task.")

    def _run_stream(self, user_input: str) -> Generator[str | Message, None, None]:
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

            self.messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in (msg.tool_calls or [])
                ],
            })

            for tc in msg.tool_calls or []:
                tool = self._tool_map.get(tc.name)
                if not tool:
                    result_str = f"ERROR: Tool '{tc.name}' not found"
                else:
                    result = tool.execute(tc)
                    result_str = result.result
                    self.events.append(AgentEvent(
                        agent_name=self.config.name,
                        event_type="tool_call",
                        data={"tool": tc.name, "args": tc.arguments, "result": result_str[:200]},
                        latency_ms=result.latency_ms,
                    ))
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        yield Message(role=Role.ASSISTANT, content="Could not complete the task.")

    def _print_summary(self):
        total_cost = sum(e.cost_usd for e in self.events)
        total_tokens = sum(e.tokens_used for e in self.events)
        total_latency = sum(e.latency_ms for e in self.events)
        tool_calls = sum(1 for e in self.events if e.event_type == "tool_call")
        llm_calls = sum(1 for e in self.events if e.event_type == "llm_call")

        print(f"\n{'='*60}")
        print(f"📊 Agent Run Summary")
        print(f"{'='*60}")
        print(f"   LLM calls:    {llm_calls}")
        print(f"   Tool calls:   {tool_calls}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Total cost:   ${total_cost:.4f}")
        print(f"   Total time:   {total_latency:.0f}ms")
        print(f"{'='*60}")
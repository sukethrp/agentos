"""AgentOS Streaming Demo — watch tokens arrive in real-time like ChatGPT."""

import sys
import time

sys.path.insert(0, "src")

from agentos.core.streaming import StreamingAgent
from agentos.tools import get_builtin_tools


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS — Streaming Demo")
    print("=" * 60)

    tools = list(get_builtin_tools().values())
    agent = StreamingAgent(
        name="stream-demo",
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools when needed. Be concise.",
        temperature=0.7,
    )

    # ── Demo 1: Simple streaming ──
    print("\n" + "━" * 60)
    print("DEMO 1: Token-by-token streaming")
    print("━" * 60)
    print()

    query = "Write a short haiku about programming."
    print(f"💬 Query: {query}")
    print("🤖 Response: ", end="", flush=True)

    start = time.time()
    token_count = 0
    for token in agent.stream_sync(query):
        print(token, end="", flush=True)
        token_count += 1

    elapsed = (time.time() - start) * 1000
    print(f"\n\n📊 Stats: {token_count} tokens | "
          f"first token: {agent.stats.first_token_ms:.0f}ms | "
          f"total: {elapsed:.0f}ms | "
          f"cost: ${agent.stats.cost_usd:.4f}")

    # ── Demo 2: Streaming with tool use ──
    print("\n" + "━" * 60)
    print("DEMO 2: Streaming with tool calls")
    print("━" * 60)
    print()

    query2 = "What's 42 * 17? Show me the answer."
    print(f"💬 Query: {query2}")
    print("🤖 Response: ", end="", flush=True)

    for token in agent.stream_sync(query2):
        print(token, end="", flush=True)

    print(f"\n\n📊 Stats: cost=${agent.stats.cost_usd:.4f} | "
          f"tool calls: {agent.stats.tool_calls} | "
          f"llm calls: {agent.stats.llm_calls}")

    # ── Demo 3: Interactive streaming chat ──
    print("\n" + "━" * 60)
    print("DEMO 3: Interactive streaming chat")
    print("━" * 60)
    print("   Type your message. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("💬 You: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break

            print("🤖 Agent: ", end="", flush=True)
            for token in agent.stream_sync(query):
                print(token, end="", flush=True)

            print(f"\n   [{agent.stats.tokens} tokens, "
                  f"${agent.stats.cost_usd:.4f}, "
                  f"{agent.stats.latency_ms:.0f}ms]\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

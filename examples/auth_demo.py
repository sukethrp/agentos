"""AgentOS Auth Demo — multi-user API keys and usage tracking."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentos.auth import create_user, default_store, usage_tracker  # type: ignore
from agentos.core.agent import Agent  # type: ignore
from agentos.tools import get_builtin_tools  # type: ignore


def divider(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    divider("🔐 AgentOS — Auth & Usage Demo")

    # ── Create or load users ──
    print("📋 Existing users:")
    users = default_store.list_users()
    for u in users:
        print(f"  • {u.email} (id={u.id}, admin={u.is_admin})")

    print("\n👤 Creating demo users (alice, bob)...")
    for email, name in [("alice@example.com", "Alice"), ("bob@example.com", "Bob")]:
        try:
            user = create_user(email=email, name=name)
            print(f"  ✅ Created {email}  api_key={user.api_key[:12]}...")
        except ValueError:
            # Already exists
            user = default_store.get_by_email(email)  # type: ignore[arg-type]
            assert user is not None
            print(f"  ↺ Using existing {email}  api_key={user.api_key[:12]}...")

    alice = default_store.get_by_email("alice@example.com")  # type: ignore[arg-type]
    bob = default_store.get_by_email("bob@example.com")  # type: ignore[arg-type]
    assert alice and bob

    divider("Step 2: Run agents as different users")

    tools = list(get_builtin_tools().values())

    alice_agent = Agent(
        name="alice-agent",
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="You are Alice's helpful personal assistant.",
    )

    bob_agent = Agent(
        name="bob-agent",
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="You are Bob's research assistant.",
    )

    # Alice runs a quick question
    print(f"🧑‍💻 Alice ({alice.email}) running a query...")
    msg = alice_agent.run("What is 12 * 7? Use the calculator tool.")
    cost = sum(e.cost_usd for e in alice_agent.events)
    tokens = sum(e.tokens_used for e in alice_agent.events)
    usage_tracker.log_usage(alice.id, tokens=tokens, cost=cost)
    print(f"   Response: {msg.content[:120]}")
    print(f"   Cost: ${cost:.4f} | Tokens: {tokens}")

    # Bob runs a different query
    print(f"\n🧑‍💻 Bob ({bob.email}) running a query...")
    msg = bob_agent.run("Give me a two-sentence summary of AgentOS.")
    cost = sum(e.cost_usd for e in bob_agent.events)
    tokens = sum(e.tokens_used for e in bob_agent.events)
    usage_tracker.log_usage(bob.id, tokens=tokens, cost=cost)
    print(f"   Response: {msg.content[:120]}")
    print(f"   Cost: ${cost:.4f} | Tokens: {tokens}")

    divider("Step 3: Usage per user")

    for user in (alice, bob):
        total = usage_tracker.get_usage(user.id).to_dict()
        day = usage_tracker.get_usage_by_period(user.id, period="day").to_dict()
        print(f"User: {user.email} (id={user.id})")
        print(
            f"  Lifetime: {total['queries']} queries, "
            f"{total['tokens']} tokens, ${total['cost']:.4f}"
        )
        print(
            f"  Last day: {day['queries']} queries, "
            f"{day['tokens']} tokens, ${day['cost']:.4f}"
        )

    print("\n" + "=" * 60)
    print("  ✅ Auth & usage demo complete")
    print("=" * 60)


"""Marketplace Demo — publish, search, install, and review agents.

Run:
    python examples/marketplace_demo.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentos.marketplace import MarketplaceStore, AgentConfig


def main():
    # Use a temp directory so we don't pollute the real store
    store = MarketplaceStore(data_dir="./agent_data/marketplace")

    print("=" * 60)
    print("🏪 AgentOS Marketplace Demo")
    print("=" * 60)

    # ── 1. Publish 3 agent templates ────────────────────────
    print("\n📦 Publishing agents...\n")

    support = store.publish(
        name="Customer Support Pro",
        description="Advanced customer support agent with ticket management, escalation, and sentiment analysis.",
        author="AgentOS Team",
        category="support",
        icon="🎧",
        tags=["support", "customer-service", "tickets"],
        price=0,
        config=AgentConfig(
            name="Customer Support Pro",
            model="gpt-4o-mini",
            system_prompt="You are a world-class customer support agent. Be empathetic, solution-oriented, and professional.",
            tools=["calculator"],
        ),
    )
    print(f"   ✓ Published: {support.name} (id={support.id})")

    analyst = store.publish(
        name="Data Analyst",
        description="Analyze datasets, create reports, find trends, and produce insights. Perfect for business intelligence.",
        author="DataWiz",
        category="data",
        icon="📊",
        tags=["data", "analytics", "reports", "bi"],
        price=29,
        config=AgentConfig(
            name="Data Analyst",
            model="gpt-4o",
            system_prompt="You are a senior data analyst. Analyze data thoroughly, present findings clearly, and provide actionable recommendations.",
            tools=["calculator", "web_search"],
        ),
    )
    print(f"   ✓ Published: {analyst.name} (id={analyst.id})")

    writer = store.publish(
        name="Content Writer",
        description="Write compelling blog posts, emails, social media content, and marketing copy.",
        author="WriterBot",
        category="writing",
        icon="✍️",
        tags=["writing", "content", "marketing", "seo"],
        price=19,
        config=AgentConfig(
            name="Content Writer",
            model="gpt-4o-mini",
            system_prompt="You are a creative content writer. Write engaging, SEO-friendly content tailored to the audience.",
            tools=[],
        ),
    )
    print(f"   ✓ Published: {writer.name} (id={writer.id})")

    # ── 2. Search the marketplace ───────────────────────────
    print("\n🔍 Searching marketplace...\n")

    results = store.search(query="data", sort_by="downloads")
    print(f"   Search 'data' → {len(results)} result(s):")
    for a in results:
        print(f"      - {a.icon} {a.name} by {a.author}")

    results = store.search(category="support")
    print(f"\n   Category 'support' → {len(results)} result(s):")
    for a in results:
        print(f"      - {a.icon} {a.name}")

    # ── 3. Install an agent ─────────────────────────────────
    print("\n📥 Installing 'Customer Support Pro'...\n")

    installed = store.install(support.id)
    if installed:
        print(f"   ✓ Installed {installed.name} — Downloads: {installed.downloads}")
        print(f"   Config: model={installed.config.model}, tools={installed.config.tools}")

    # Install again to bump the counter
    store.install(support.id)
    store.install(analyst.id)

    # ── 4. Leave reviews ────────────────────────────────────
    print("\n⭐ Leaving reviews...\n")

    r1 = store.review(support.id, user="alice", rating=5, comment="Amazing support agent! Reduced our ticket backlog by 40%.")
    print(f"   ✓ alice reviewed {support.name}: {r1.rating}/5 — \"{r1.comment}\"")

    r2 = store.review(support.id, user="bob", rating=4, comment="Great, but could use better escalation logic.")
    print(f"   ✓ bob reviewed {support.name}: {r2.rating}/5 — \"{r2.comment}\"")

    r3 = store.review(analyst.id, user="charlie", rating=5, comment="Best data agent I've used. Worth every penny!")
    print(f"   ✓ charlie reviewed {analyst.name}: {r3.rating}/5 — \"{r3.comment}\"")

    # ── 5. Show trending & top-rated ────────────────────────
    print("\n🔥 Trending Agents (most downloads):\n")
    for a in store.get_trending():
        print(f"   {a.icon} {a.name} — ↓{a.downloads} downloads, ★{a.rating}")

    print("\n🏆 Top Rated Agents:\n")
    for a in store.get_top_rated():
        print(f"   {a.icon} {a.name} — ★{a.rating} ({a.review_count} reviews)")

    # ── 6. Stats ────────────────────────────────────────────
    stats = store.stats()
    print("\n📈 Marketplace Stats:\n")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("✅ Marketplace demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

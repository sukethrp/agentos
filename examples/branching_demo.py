"""Conversation Branching Demo — explore "what if" scenarios with AgentOS.

This demo shows how to:
1. Start a conversation about investment strategy
2. Branch at a decision point
3. Explore "aggressive" strategy in Branch A
4. Explore "conservative" strategy in Branch B
5. Compare the two paths side-by-side
6. Merge insights from both branches
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentos.core.branching import ConversationTree


def main():
    print("=" * 60)
    print("🌿 AgentOS Conversation Branching Demo")
    print("=" * 60)

    # ── Step 1: Create a conversation tree ──
    tree = ConversationTree()
    print(f"\n📌 Created conversation tree: {tree.tree_id}")
    print(f"   Main branch: {tree.main_branch_id}")

    # ── Step 2: Build a conversation ──
    print("\n── Building initial conversation ──")
    tree.add_message("user", "I have $100,000 to invest. What should I do?")
    tree.add_message("assistant",
        "Great question! There are several approaches depending on your goals:\n"
        "1. Risk tolerance: How much volatility can you handle?\n"
        "2. Time horizon: When do you need the money?\n"
        "3. Income needs: Do you need regular income from investments?\n\n"
        "What's your risk tolerance — aggressive, moderate, or conservative?"
    )
    tree.add_message("user", "I'm 30 years old with a stable job. I have 20+ years before retirement.")
    tree.add_message("assistant",
        "With a 20+ year horizon, you have time to recover from market downturns. "
        "This opens up both aggressive and conservative paths:\n\n"
        "**Option A - Aggressive Growth:** 80% stocks (tech/growth), 15% international, 5% crypto\n"
        "**Option B - Conservative Balance:** 60% index funds, 30% bonds, 10% REITs\n\n"
        "Which direction interests you more?"
    )

    print(f"   Added {tree.active_branch.message_count} messages to main branch")
    for m in tree.get_messages():
        preview = m.content[:60].replace("\n", " ")
        print(f"   [{m.index}] {m.role}: {preview}...")

    # ── Step 3: Branch at the decision point (message index 3) ──
    print("\n── Branching at message #3 (the decision point) ──")

    # Create the aggressive branch
    aggressive_id = tree.branch(at_message_index=3, label="aggressive")
    print(f"   Created branch '{aggressive_id}' (aggressive)")

    # Create the conservative branch
    conservative_id = tree.branch(at_message_index=3, label="conservative")
    print(f"   Created branch '{conservative_id}' (conservative)")

    # ── Step 4: Explore the aggressive path ──
    print("\n── Exploring AGGRESSIVE strategy ──")
    tree.switch_branch(aggressive_id)

    tree.add_message("user", "Let's go aggressive! I want maximum growth.")
    tree.add_message("assistant",
        "Here's your Aggressive Growth Portfolio:\n\n"
        "• $40,000 — Growth ETFs (VUG, QQQ) — tech-heavy large caps\n"
        "• $20,000 — Individual tech stocks (diversified across 5-8 companies)\n"
        "• $15,000 — International growth (VWO, IXUS)\n"
        "• $10,000 — Small-cap growth funds (VBK)\n"
        "• $10,000 — Cryptocurrency (70% BTC, 30% ETH)\n"
        "• $5,000 — Cash reserve for opportunities\n\n"
        "Expected annual return: 10-14% (with high volatility)\n"
        "Risk: High — could lose 30-40% in a downturn\n"
        "Best for: Long time horizon, high risk tolerance"
    )
    tree.add_message("user", "What's the worst case scenario?")
    tree.add_message("assistant",
        "Worst case scenarios for the aggressive portfolio:\n\n"
        "• 2008-level crash: Portfolio could drop ~45% ($55K loss)\n"
        "• Crypto winter: 10% allocation could lose 80% ($8K)\n"
        "• Tech bubble: Growth stocks could underperform for 3-5 years\n"
        "• Recovery time: Historically 3-5 years to recover from major crashes\n\n"
        "However, with 20+ years, you'd likely recover and grow significantly. "
        "The S&P 500 has returned ~10% annually over any 20-year period in history."
    )

    print(f"   Branch now has {tree.active_branch.message_count} messages")

    # ── Step 5: Explore the conservative path ──
    print("\n── Exploring CONSERVATIVE strategy ──")
    tree.switch_branch(conservative_id)

    tree.add_message("user", "I'd prefer a safer approach. Let's be conservative.")
    tree.add_message("assistant",
        "Here's your Conservative Balance Portfolio:\n\n"
        "• $35,000 — Total stock market index (VTI) — broad diversification\n"
        "• $25,000 — Total bond market (BND) — stability and income\n"
        "• $15,000 — International developed markets (VXUS)\n"
        "• $10,000 — REITs (VNQ) — real estate exposure and dividends\n"
        "• $10,000 — Treasury I-Bonds — inflation protection\n"
        "• $5,000 — High-yield savings — emergency fund\n\n"
        "Expected annual return: 6-8%\n"
        "Risk: Moderate-low — max drawdown ~15-20%\n"
        "Best for: Steady growth with less stress"
    )
    tree.add_message("user", "How much would I have in 20 years?")
    tree.add_message("assistant",
        "Projected growth over 20 years (no additional contributions):\n\n"
        "• At 6% annually: $100K → $320,714\n"
        "• At 7% annually: $100K → $386,968\n"
        "• At 8% annually: $100K → $466,096\n\n"
        "If you add $500/month ($6K/year):\n"
        "• At 7%: $100K + contributions → $648,000+\n\n"
        "The power of compounding! Even conservative returns build significant "
        "wealth over 20 years. And you'll sleep better at night."
    )

    print(f"   Branch now has {tree.active_branch.message_count} messages")

    # ── Step 6: List all branches ──
    print("\n── All Branches ──")
    for b in tree.list_branches():
        active = " ← ACTIVE" if b["is_active"] else ""
        main = " (main)" if b["is_main"] else ""
        print(f"   {b['label']}{main}: {b['message_count']} messages · ID: {b['branch_id']}{active}")

    # ── Step 7: Compare the two branches ──
    print("\n── Comparing aggressive vs conservative ──")
    comparison = tree.compare_branches(aggressive_id, conservative_id)

    print(f"   Shared messages: {comparison['shared_count']}")
    print(f"   Aggressive unique: {len(comparison['branch_a_unique'])} messages")
    print(f"   Conservative unique: {len(comparison['branch_b_unique'])} messages")

    print("\n   AGGRESSIVE path explored:")
    for m in comparison["branch_a_unique"]:
        preview = m["content"][:70].replace("\n", " ")
        print(f"      [{m['role']}] {preview}...")

    print("\n   CONSERVATIVE path explored:")
    for m in comparison["branch_b_unique"]:
        preview = m["content"][:70].replace("\n", " ")
        print(f"      [{m['role']}] {preview}...")

    # ── Step 8: Merge both branches ──
    print("\n── Merging both paths ──")
    merged_id = tree.merge_branches(aggressive_id, conservative_id, label="balanced-merge")

    tree.switch_branch(merged_id)
    print(f"   Merged branch: {merged_id}")
    print(f"   Contains {tree.active_branch.message_count} messages (shared + merge context)")

    # Show the merge summary
    merge_msg = tree.get_messages()[-1]
    if merge_msg.role == "system":
        print("\n   Merge context preview:")
        for line in merge_msg.content.split("\n")[:8]:
            print(f"      {line}")
        print("      ...")

    # ── Step 9: Continue on merged branch ──
    print("\n── Continuing on merged branch ──")
    tree.add_message("user",
        "Given both the aggressive and conservative approaches, "
        "what would you recommend as a balanced strategy?"
    )
    print(f"   Added follow-up question. Branch now has {tree.active_branch.message_count} messages.")
    print("   (In production, the agent would respond using context from both paths)")

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("📊 Final Tree Summary")
    print("=" * 60)
    tree_info = tree.to_dict()
    print(f"   Tree ID: {tree_info['tree_id']}")
    print(f"   Branches: {tree_info['branch_count']}")
    for b in tree_info["branches"]:
        active = " ← active" if b["is_active"] else ""
        print(f"      {b['label']}: {b['message_count']} msgs{active}")
    print("=" * 60)
    print("\n✅ Demo complete! The branching system lets you explore")
    print("   multiple conversation paths and merge insights together.\n")


if __name__ == "__main__":
    main()

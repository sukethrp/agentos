#!/usr/bin/env python3
"""
Learning System Demo — show how agents improve from user feedback.

This demo:
1. Seeds realistic feedback (thumbs, ratings, corrections) across topics
2. Analyses patterns ("refund questions fail 40% of the time")
3. Auto-generates prompt patches for weak areas
4. Builds few-shot examples from the best interactions
5. Shows a learning-progress report with improvement trends

Run:
    python examples/learning_demo.py
"""

from __future__ import annotations

import json
import random
import textwrap
import time

from agentos.learning.feedback import FeedbackStore, FeedbackType
from agentos.learning.analyzer import FeedbackAnalyzer
from agentos.learning.prompt_optimizer import PromptOptimizer
from agentos.learning.few_shot import FewShotBuilder
from agentos.learning.report import build_learning_report


DIVIDER = "═" * 60


def pp(label: str, obj: dict | list | str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(DIVIDER)
    if isinstance(obj, (dict, list)):
        print(textwrap.indent(json.dumps(obj, indent=2, default=str)[:1200], "  "))
    else:
        print(textwrap.indent(str(obj)[:1200], "  "))


def main() -> None:
    print("🧠 AgentOS Learning System Demo")
    print("=" * 60)

    # Use in-memory store (temp dir) so we don't pollute the real data
    import tempfile, os
    tmp = tempfile.mkdtemp()
    store = FeedbackStore(data_dir=tmp)

    # ── 1. Seed realistic feedback ───────────────────────────────────────

    print("\n📝 STEP 1: Seeding feedback data…")

    base_ts = time.time() - 21 * 86400  # 3 weeks ago

    # Week 1 — agent is mediocre
    week1_feedback = [
        ("How do I get a refund?", "Please contact support.", "thumbs_down", "refund",
         "You should explain the refund policy and timeline"),
        ("Cancel my subscription", "I'll help with that.", "rating_2", "refund", ""),
        ("I want my money back NOW", "We apologize for the inconvenience.", "thumbs_down", "refund",
         "Say: We'll process your refund within 5-7 business days. Here's how…"),
        ("What are your pricing plans?", "We have Free, Pro, and Enterprise tiers.", "thumbs_up", "billing", ""),
        ("How much does Pro cost?", "Pro is $29/month with 50K requests.", "rating_5", "billing", ""),
        ("My login isn't working", "Try resetting your password at /reset.", "thumbs_up", "account", ""),
        ("I keep getting error 500", "Sorry about that. Can you try again?", "thumbs_down", "technical",
         "Ask for the error details, check the status page, and provide specific troubleshooting steps"),
        ("How do I set up the API?", "Check our docs at docs.example.com.", "rating_3", "integration", ""),
        ("I'm new, where do I start?", "Welcome! Start at the dashboard.", "thumbs_up", "onboarding", ""),
        ("Your product is terrible!", "We're sorry you feel that way.", "thumbs_down", "general", ""),
    ]

    from agentos.learning.feedback import FeedbackEntry

    def _seed(rows, time_offset=0):
        for i, (query, response, fb_type, topic, correction) in enumerate(rows):
            ts = base_ts + time_offset + i * 3600 + random.uniform(0, 1800)
            if fb_type == "thumbs_up":
                store.add(FeedbackEntry(
                    feedback_type=FeedbackType.THUMBS_UP, query=query, response=response,
                    agent_name="support-bot", topic=topic, timestamp=ts,
                ))
            elif fb_type == "thumbs_down":
                store.add(FeedbackEntry(
                    feedback_type=FeedbackType.THUMBS_DOWN, query=query, response=response,
                    agent_name="support-bot", topic=topic, timestamp=ts,
                    comment="Not helpful" if not correction else "",
                ))
                if correction:
                    store.add(FeedbackEntry(
                        feedback_type=FeedbackType.CORRECTION, query=query, response=response,
                        correction=correction, agent_name="support-bot", topic=topic, timestamp=ts + 60,
                    ))
            elif fb_type.startswith("rating_"):
                rating = float(fb_type.split("_")[1])
                store.add(FeedbackEntry(
                    feedback_type=FeedbackType.RATING, query=query, response=response,
                    rating=rating, agent_name="support-bot", topic=topic, timestamp=ts,
                ))

    _seed(week1_feedback, time_offset=0)

    # Week 2 — slightly improved (after hypothetical first round of patches)
    week2_feedback = [
        ("Process my refund please", "I'll initiate your refund. It takes 5-7 business days.", "thumbs_up", "refund", ""),
        ("Cancel and refund", "I understand. Let me process your cancellation and refund right away.", "rating_4", "refund", ""),
        ("Refund my last payment", "Sure! Your refund will be processed within 5-7 days.", "thumbs_up", "refund", ""),
        ("What plan should I choose?", "It depends on your usage. Free for <1K req, Pro for up to 50K.", "rating_5", "billing", ""),
        ("API rate limits?", "Free: 100/min, Pro: 1000/min, Enterprise: custom.", "thumbs_up", "integration", ""),
        ("Getting 403 errors", "That's usually an auth issue. Check your API key and permissions.", "rating_4", "technical", ""),
        ("How do I connect Slack?", "Go to Settings → Integrations → Add Slack.", "thumbs_up", "integration", ""),
        ("Password reset not working", "Sorry about that. Try clearing cookies first, then visit /reset.", "thumbs_up", "account", ""),
        ("Your support is slow!", "I apologize. Let me help you right now.", "rating_3", "general", ""),
    ]

    _seed(week2_feedback, time_offset=7 * 86400)

    # Week 3 — further improved
    week3_feedback = [
        ("I need a refund for order #123", "I've located order #123. Your refund of $49.99 will be processed in 3-5 days.", "rating_5", "refund", ""),
        ("Enterprise pricing?", "Enterprise starts at $499/mo with custom SLA. Let me schedule a call.", "thumbs_up", "billing", ""),
        ("Error when uploading files", "What file format and size? Check our limits at docs.example.com/uploads.", "thumbs_up", "technical", ""),
        ("Set up SSO for our team", "Great! Go to Admin → Security → SSO. We support SAML and OAuth.", "rating_5", "integration", ""),
        ("I'm confused about the dashboard", "No problem! The dashboard has 3 sections: Overview, Analytics, Settings. Let me walk you through each.", "thumbs_up", "onboarding", ""),
    ]

    _seed(week3_feedback, time_offset=14 * 86400)

    stats = store.stats()
    print(f"  Total feedback entries: {stats['total']}")
    print(f"  Positive: {stats['positive']}  Negative: {stats['negative']}")
    print(f"  Positive rate: {stats['positive_rate']}%")
    print(f"  Corrections: {stats['corrections']}")

    # ── 2. Analyze patterns ──────────────────────────────────────────────

    print("\n\n🔍 STEP 2: Analyzing feedback patterns…")

    analyzer = FeedbackAnalyzer(store)
    analysis = analyzer.analyze()
    print(analysis.summary_text())

    # ── 3. Generate prompt patches ───────────────────────────────────────

    print("\n\n🔧 STEP 3: Generating prompt optimisations…")

    optimizer = PromptOptimizer(store, use_llm=False)
    patches = optimizer.optimize()

    print(f"  Generated {len(patches)} prompt patches:")
    for p in patches:
        print(f"\n  📌 [{p.topic}] (confidence={p.confidence:.0%}, source={p.source})")
        for line in p.instruction.split("\n")[:4]:
            print(f"     {line}")
        if len(p.instruction.split("\n")) > 4:
            print(f"     …")

    # Show the optimised prompt
    base_prompt = "You are a helpful customer support assistant."
    optimised = optimizer.apply_patches(base_prompt)
    pp("Optimised System Prompt", optimised)

    # ── 4. Build few-shot examples ───────────────────────────────────────

    print("\n\n📚 STEP 4: Building few-shot examples…")

    builder = FewShotBuilder(store, max_examples=4, max_per_topic=2)
    examples = builder.build()

    print(f"  Selected {len(examples)} examples:")
    for ex in examples:
        print(f"\n  [{ex.topic}] (source={ex.source}, quality={ex.quality_score:.1f})")
        print(f"    User: {ex.query[:80]}")
        print(f"    Asst: {ex.response[:100]}…" if len(ex.response) > 100 else f"    Asst: {ex.response}")

    few_shot_section = builder.to_prompt_section()
    if few_shot_section:
        pp("Few-Shot Prompt Section", few_shot_section)

    pp("Few-Shot Stats", builder.stats())

    # ── 5. Learning progress report ──────────────────────────────────────

    print("\n\n📈 STEP 5: Learning progress report…")

    report = build_learning_report(store, period="week")
    print(report.summary_text())

    pp("Quality Chart Data", report.quality_chart)
    pp("Topic Trends", [t.to_dict() for t in report.topic_trends])

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("✅ Learning System Demo Complete!")
    print(f"   Total feedback: {stats['total']}")
    print(f"   Prompt patches generated: {len(patches)}")
    print(f"   Few-shot examples built: {len(examples)}")
    print(f"   Quality trend: {report.direction} ({report.quality_change:+.2f})")
    print(f"   Improving topics: {', '.join(report.improving_topics) or 'N/A'}")
    print(f"   Declining topics: {', '.join(report.declining_topics) or 'none'}")
    print(f"{'=' * 60}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

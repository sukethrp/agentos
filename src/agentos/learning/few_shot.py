"""Few-Shot Builder — curate dynamic few-shot examples from top-rated interactions.

Instead of hand-writing few-shot examples, the builder automatically
selects the *best* real interactions from feedback and formats them as
example messages that get injected into the system prompt.

Selection criteria:
1. Highest-rated interactions (thumbs-up or ≥4 stars)
2. Covers diverse topics (not all from the same category)
3. Includes any user-corrected examples (gold standard)
4. Prioritises shorter, crisper examples over long ones

The output is a list of ``{"role": "user", ...}, {"role": "assistant", ...}``
pairs ready to splice into the message list.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from agentos.learning.analyzer import detect_topic
from agentos.learning.feedback import FeedbackEntry, FeedbackStore, FeedbackType


@dataclass
class FewShotExample:
    """One user→assistant example pair."""

    topic: str = ""
    query: str = ""
    response: str = ""
    source: str = ""           # "positive_feedback" | "correction" | "manual"
    quality_score: float = 0.0

    def to_messages(self) -> list[dict]:
        return [
            {"role": "user", "content": self.query},
            {"role": "assistant", "content": self.response},
        ]

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "query": self.query[:120],
            "response": self.response[:200],
            "source": self.source,
            "quality_score": round(self.quality_score, 1),
        }


class FewShotBuilder:
    """Build a set of few-shot examples from feedback data."""

    def __init__(
        self,
        store: FeedbackStore | None = None,
        max_examples: int = 6,
        max_per_topic: int = 2,
        max_response_length: int = 500,
    ) -> None:
        from agentos.learning.feedback import get_feedback_store
        self.store = store or get_feedback_store()
        self.max_examples = max_examples
        self.max_per_topic = max_per_topic
        self.max_response_length = max_response_length
        self._examples: list[FewShotExample] = []

    # ── Build ────────────────────────────────────────────────────────────

    def build(self) -> list[FewShotExample]:
        """Select the best few-shot examples from the feedback store."""
        candidates: list[FewShotExample] = []

        # 1. Corrections are gold-standard — they show the *right* answer
        for entry in self.store.corrections():
            if entry.correction.strip():
                candidates.append(FewShotExample(
                    topic=entry.topic or detect_topic(entry.query),
                    query=entry.query,
                    response=entry.correction,
                    source="correction",
                    quality_score=9.5,
                ))

        # 2. Highly-rated positive interactions
        for entry in self.store.positive():
            if entry.feedback_type == FeedbackType.THUMBS_UP or (
                entry.feedback_type == FeedbackType.RATING and entry.rating >= 4
            ):
                resp = entry.response.strip()
                if not resp or len(resp) > self.max_response_length * 2:
                    continue
                candidates.append(FewShotExample(
                    topic=entry.topic or detect_topic(entry.query),
                    query=entry.query,
                    response=resp,
                    source="positive_feedback",
                    quality_score=entry.quality_score,
                ))

        if not candidates:
            self._examples = []
            return []

        # 3. Deduplicate by query (keep highest quality)
        seen: dict[str, FewShotExample] = {}
        for c in candidates:
            key = c.query.strip().lower()[:80]
            if key not in seen or c.quality_score > seen[key].quality_score:
                seen[key] = c
        candidates = list(seen.values())

        # 4. Topic-diverse selection
        selected = self._diverse_select(candidates)

        # 5. Trim long responses
        for ex in selected:
            if len(ex.response) > self.max_response_length:
                ex.response = ex.response[: self.max_response_length].rsplit(" ", 1)[0] + "…"

        self._examples = selected
        return selected

    def _diverse_select(self, candidates: list[FewShotExample]) -> list[FewShotExample]:
        """Select up to *max_examples*, limiting *max_per_topic* per topic."""
        by_topic: dict[str, list[FewShotExample]] = defaultdict(list)
        for c in candidates:
            by_topic[c.topic].append(c)

        # Sort each topic bucket by quality descending
        for topic in by_topic:
            by_topic[topic].sort(key=lambda x: -x.quality_score)

        selected: list[FewShotExample] = []
        topic_counts: dict[str, int] = defaultdict(int)

        # Round-robin across topics, picking the best from each
        topics = list(by_topic.keys())
        random.shuffle(topics)
        changed = True
        while changed and len(selected) < self.max_examples:
            changed = False
            for topic in topics:
                if len(selected) >= self.max_examples:
                    break
                if topic_counts[topic] >= self.max_per_topic:
                    continue
                bucket = by_topic[topic]
                idx = topic_counts[topic]
                if idx < len(bucket):
                    selected.append(bucket[idx])
                    topic_counts[topic] += 1
                    changed = True

        return selected

    # ── Output formats ───────────────────────────────────────────────────

    def to_messages(self) -> list[dict]:
        """Return flat list of message dicts ready for the OpenAI API."""
        msgs: list[dict] = []
        for ex in self._examples:
            msgs.extend(ex.to_messages())
        return msgs

    def to_prompt_section(self) -> str:
        """Return a text block for injection into the system prompt."""
        if not self._examples:
            return ""

        lines = ["# Examples of Good Responses", ""]
        for i, ex in enumerate(self._examples, 1):
            lines.append(f"**Example {i}** ({ex.topic})")
            lines.append(f"User: {ex.query}")
            lines.append(f"Assistant: {ex.response}")
            lines.append("")

        return "\n".join(lines)

    def inject_into_messages(
        self, system_prompt: str, user_input: str,
    ) -> list[dict]:
        """Build a message list with few-shot examples after the system prompt."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.to_messages())
        messages.append({"role": "user", "content": user_input})
        return messages

    def get_examples(self) -> list[dict]:
        return [e.to_dict() for e in self._examples]

    def stats(self) -> dict:
        topics = defaultdict(int)
        sources = defaultdict(int)
        for e in self._examples:
            topics[e.topic] += 1
            sources[e.source] += 1
        return {
            "total_examples": len(self._examples),
            "topics_covered": dict(topics),
            "sources": dict(sources),
            "avg_quality": (
                round(sum(e.quality_score for e in self._examples) / len(self._examples), 1)
                if self._examples else 0
            ),
        }

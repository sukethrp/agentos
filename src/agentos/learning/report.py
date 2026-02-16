"""Learning Progress Report — track improvement over time.

Compares quality metrics across time windows to answer:
- "Is the agent getting better?"
- "Which topics improved after prompt optimisation?"
- "What is the week-over-week quality trend?"
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from agentos.learning.feedback import FeedbackEntry, FeedbackStore


DAY = 86_400
WEEK = 7 * DAY


@dataclass
class PeriodSnapshot:
    """Aggregated quality metrics for one time period."""

    label: str
    start: float
    end: float
    total: int = 0
    positive: int = 0
    negative: int = 0
    avg_quality: float = 0.0
    positive_rate: float = 0.0
    corrections: int = 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total": self.total,
            "positive": self.positive,
            "negative": self.negative,
            "avg_quality": round(self.avg_quality, 2),
            "positive_rate": round(self.positive_rate, 1),
            "corrections": self.corrections,
        }


@dataclass
class TopicTrend:
    """Quality trend for a single topic across time periods."""

    topic: str
    periods: list[PeriodSnapshot] = field(default_factory=list)
    overall_avg: float = 0.0
    direction: str = "stable"  # "improving" | "declining" | "stable"
    change_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "periods": [p.to_dict() for p in self.periods],
            "overall_avg": round(self.overall_avg, 2),
            "direction": self.direction,
            "change_pct": round(self.change_pct, 1),
        }


@dataclass
class LearningReport:
    """Comprehensive improvement-over-time report."""

    generated_at: float = 0.0
    total_feedback: int = 0
    current_avg_quality: float = 0.0
    previous_avg_quality: float = 0.0
    quality_change: float = 0.0
    direction: str = "stable"

    timeline: list[PeriodSnapshot] = field(default_factory=list)
    topic_trends: list[TopicTrend] = field(default_factory=list)
    improving_topics: list[str] = field(default_factory=list)
    declining_topics: list[str] = field(default_factory=list)
    stable_topics: list[str] = field(default_factory=list)

    # Chart-ready data
    quality_chart: list[dict] = field(default_factory=list)
    positive_rate_chart: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_feedback": self.total_feedback,
            "current_avg_quality": round(self.current_avg_quality, 2),
            "previous_avg_quality": round(self.previous_avg_quality, 2),
            "quality_change": round(self.quality_change, 2),
            "direction": self.direction,
            "timeline": [p.to_dict() for p in self.timeline],
            "topic_trends": [t.to_dict() for t in self.topic_trends],
            "improving_topics": self.improving_topics,
            "declining_topics": self.declining_topics,
            "stable_topics": self.stable_topics,
            "quality_chart": self.quality_chart,
            "positive_rate_chart": self.positive_rate_chart,
        }

    def summary_text(self) -> str:
        arrow = "↑" if self.direction == "improving" else ("↓" if self.direction == "declining" else "→")
        lines = [
            "=" * 60,
            "  LEARNING PROGRESS REPORT",
            "=" * 60,
            f"  Total feedback     : {self.total_feedback}",
            f"  Current quality    : {self.current_avg_quality:.2f} / 10",
            f"  Previous quality   : {self.previous_avg_quality:.2f} / 10",
            f"  Change             : {self.quality_change:+.2f}  {arrow}  ({self.direction})",
        ]
        if self.improving_topics:
            lines.append(f"\n  📈 Improving: {', '.join(self.improving_topics)}")
        if self.declining_topics:
            lines.append(f"  📉 Declining: {', '.join(self.declining_topics)}")

        lines.append("\n  Weekly timeline:")
        for p in self.timeline:
            bar = "█" * max(int(p.avg_quality), 1)
            lines.append(
                f"    {p.label:<10}  quality={p.avg_quality:.1f}  "
                f"rate={p.positive_rate:.0f}%  {bar}  n={p.total}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Builder ──────────────────────────────────────────────────────────────────

def _snapshot(label: str, start: float, end: float, entries: list[FeedbackEntry]) -> PeriodSnapshot:
    bucket = [e for e in entries if start <= e.timestamp < end]
    ps = PeriodSnapshot(label=label, start=start, end=end)
    ps.total = len(bucket)
    if not bucket:
        return ps
    ps.positive = sum(1 for e in bucket if e.is_positive)
    ps.negative = ps.total - ps.positive
    ps.avg_quality = statistics.mean(e.quality_score for e in bucket)
    ps.positive_rate = ps.positive / ps.total * 100
    ps.corrections = sum(1 for e in bucket if e.correction)
    return ps


def build_learning_report(
    store: FeedbackStore | None = None,
    period: str = "week",
) -> LearningReport:
    """Build a learning progress report from feedback data."""
    from agentos.learning.feedback import get_feedback_store
    from agentos.learning.analyzer import auto_tag_topics

    store = store or get_feedback_store()
    entries = auto_tag_topics(store.all())

    report = LearningReport(generated_at=time.time())
    report.total_feedback = len(entries)
    if not entries:
        return report

    # Determine period length
    period_len = WEEK if period == "week" else DAY
    oldest = min(e.timestamp for e in entries)
    newest = max(e.timestamp for e in entries)

    # Build timeline snapshots
    cursor = oldest
    period_num = 1
    snapshots: list[PeriodSnapshot] = []
    while cursor <= newest + period_len:
        end = cursor + period_len
        label = f"{'W' if period == 'week' else 'D'}{period_num}"
        snap = _snapshot(label, cursor, end, entries)
        if snap.total > 0:
            snapshots.append(snap)
        cursor = end
        period_num += 1

    report.timeline = snapshots

    # Current vs previous period quality
    if len(snapshots) >= 2:
        report.current_avg_quality = snapshots[-1].avg_quality
        report.previous_avg_quality = snapshots[-2].avg_quality
        report.quality_change = report.current_avg_quality - report.previous_avg_quality
        if report.quality_change > 0.3:
            report.direction = "improving"
        elif report.quality_change < -0.3:
            report.direction = "declining"
        else:
            report.direction = "stable"
    elif snapshots:
        report.current_avg_quality = snapshots[-1].avg_quality
        report.previous_avg_quality = snapshots[-1].avg_quality

    # Chart data
    report.quality_chart = [
        {"label": s.label, "avg_quality": round(s.avg_quality, 2)} for s in snapshots
    ]
    report.positive_rate_chart = [
        {"label": s.label, "positive_rate": round(s.positive_rate, 1)} for s in snapshots
    ]

    # Per-topic trends
    topics: set[str] = {e.topic for e in entries if e.topic}
    for topic in topics:
        topic_entries = [e for e in entries if e.topic == topic]
        if len(topic_entries) < 2:
            continue

        periods: list[PeriodSnapshot] = []
        cursor = oldest
        pn = 1
        while cursor <= newest + period_len:
            end = cursor + period_len
            label = f"{'W' if period == 'week' else 'D'}{pn}"
            snap = _snapshot(label, cursor, end, topic_entries)
            if snap.total > 0:
                periods.append(snap)
            cursor = end
            pn += 1

        tt = TopicTrend(topic=topic, periods=periods)
        tt.overall_avg = statistics.mean(e.quality_score for e in topic_entries)

        if len(periods) >= 2:
            first_q = periods[0].avg_quality
            last_q = periods[-1].avg_quality
            tt.change_pct = ((last_q - first_q) / max(first_q, 0.1)) * 100
            if tt.change_pct > 10:
                tt.direction = "improving"
            elif tt.change_pct < -10:
                tt.direction = "declining"
            else:
                tt.direction = "stable"

        report.topic_trends.append(tt)

    report.topic_trends.sort(key=lambda t: -t.overall_avg)
    report.improving_topics = [t.topic for t in report.topic_trends if t.direction == "improving"]
    report.declining_topics = [t.topic for t in report.topic_trends if t.direction == "declining"]
    report.stable_topics = [t.topic for t in report.topic_trends if t.direction == "stable"]

    return report

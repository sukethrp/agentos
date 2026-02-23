"""Feedback Analyzer — discover patterns in user feedback.

Answers questions like:
- "Which topics does the agent fail on most often?"
- "Which tools cause problems?"
- "What time periods had the worst feedback?"
- "Are refund questions getting worse over time?"

All analysis is done with pure Python — no pandas or numpy required.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from agentos.learning.feedback import FeedbackEntry, FeedbackStore, FeedbackType


# ── Topic detection ──────────────────────────────────────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "billing": [
        "bill",
        "invoice",
        "charge",
        "payment",
        "subscription",
        "pricing",
        "price",
        "cost",
        "plan",
        "tier",
    ],
    "refund": ["refund", "money back", "cancel", "cancellation", "return"],
    "technical": [
        "error",
        "bug",
        "crash",
        "broken",
        "not working",
        "issue",
        "fail",
        "exception",
    ],
    "account": [
        "login",
        "password",
        "account",
        "sign in",
        "sign up",
        "register",
        "profile",
        "sso",
        "auth",
    ],
    "onboarding": [
        "getting started",
        "setup",
        "install",
        "new user",
        "beginner",
        "tutorial",
        "how to",
    ],
    "integration": ["api", "webhook", "integrate", "sdk", "endpoint", "connect"],
    "performance": ["slow", "latency", "timeout", "speed", "performance"],
    "security": ["security", "gdpr", "compliance", "soc2", "encrypt", "data privacy"],
    "feature_request": [
        "feature",
        "wish",
        "would be nice",
        "could you add",
        "suggestion",
    ],
    "general": [],
}


def detect_topic(query: str) -> str:
    """Simple keyword-based topic detection."""
    q = query.lower()
    scores: dict[str, int] = {}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in q)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else "general"


def auto_tag_topics(entries: list[FeedbackEntry]) -> list[FeedbackEntry]:
    """Backfill ``.topic`` on entries that don't have one."""
    for e in entries:
        if not e.topic:
            e.topic = detect_topic(e.query)
    return entries


# ── Analysis results ─────────────────────────────────────────────────────────


@dataclass
class TopicAnalysis:
    topic: str
    total: int = 0
    positive: int = 0
    negative: int = 0
    avg_quality: float = 0.0
    failure_rate: float = 0.0
    sample_queries: list[str] = field(default_factory=list)
    common_complaints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "total": self.total,
            "positive": self.positive,
            "negative": self.negative,
            "avg_quality": round(self.avg_quality, 2),
            "failure_rate": round(self.failure_rate, 1),
            "sample_queries": self.sample_queries[:5],
            "common_complaints": self.common_complaints[:5],
        }


@dataclass
class ToolAnalysis:
    tool_name: str
    total_uses: int = 0
    positive_when_used: int = 0
    negative_when_used: int = 0
    failure_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "total_uses": self.total_uses,
            "positive_when_used": self.positive_when_used,
            "negative_when_used": self.negative_when_used,
            "failure_rate": round(self.failure_rate, 1),
        }


@dataclass
class TimeWindow:
    label: str
    start: float
    end: float
    total: int = 0
    positive: int = 0
    avg_quality: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total": self.total,
            "positive": self.positive,
            "avg_quality": round(self.avg_quality, 2),
            "positive_rate": round(self.positive / max(self.total, 1) * 100, 1),
        }


@dataclass
class AnalysisReport:
    """Full analysis of feedback patterns."""

    total_feedback: int = 0
    positive_rate: float = 0.0
    avg_quality: float = 0.0
    topics: list[TopicAnalysis] = field(default_factory=list)
    worst_topics: list[str] = field(default_factory=list)
    best_topics: list[str] = field(default_factory=list)
    tools: list[ToolAnalysis] = field(default_factory=list)
    worst_tools: list[str] = field(default_factory=list)
    time_trend: list[TimeWindow] = field(default_factory=list)
    trending_issues: list[str] = field(default_factory=list)
    top_corrections: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_feedback": self.total_feedback,
            "positive_rate": round(self.positive_rate, 1),
            "avg_quality": round(self.avg_quality, 2),
            "topics": [t.to_dict() for t in self.topics],
            "worst_topics": self.worst_topics,
            "best_topics": self.best_topics,
            "tools": [t.to_dict() for t in self.tools],
            "worst_tools": self.worst_tools,
            "time_trend": [tw.to_dict() for tw in self.time_trend],
            "trending_issues": self.trending_issues,
            "top_corrections": self.top_corrections[:10],
        }

    def summary_text(self) -> str:
        lines = [
            "=" * 60,
            "  FEEDBACK ANALYSIS",
            "=" * 60,
            f"  Total feedback  : {self.total_feedback}",
            f"  Positive rate   : {self.positive_rate:.1f}%",
            f"  Avg quality     : {self.avg_quality:.2f} / 10",
        ]
        if self.worst_topics:
            lines.append(f"\n  ⚠️  Weakest topics: {', '.join(self.worst_topics)}")
        if self.best_topics:
            lines.append(f"  ✅ Strongest topics: {', '.join(self.best_topics)}")
        if self.worst_tools:
            lines.append(f"  🔧 Problematic tools: {', '.join(self.worst_tools)}")
        if self.trending_issues:
            lines.append("\n  📈 Trending issues:")
            for issue in self.trending_issues[:5]:
                lines.append(f"     • {issue}")

        lines.append("\n  Per-topic breakdown:")
        for t in self.topics:
            bar = "█" * max(int(t.avg_quality), 1)
            lines.append(
                f"    {t.topic:<16} fail={t.failure_rate:5.1f}%  "
                f"quality={t.avg_quality:.1f}  {bar}  n={t.total}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Analyzer ─────────────────────────────────────────────────────────────────


class FeedbackAnalyzer:
    """Analyze a set of feedback entries and produce pattern insights."""

    def __init__(self, store: FeedbackStore | None = None) -> None:
        from agentos.learning.feedback import get_feedback_store

        self.store = store or get_feedback_store()

    def analyze(self, entries: list[FeedbackEntry] | None = None) -> AnalysisReport:
        """Run a full analysis.  Pass *entries* to analyze a subset."""
        entries = entries or self.store.all()
        entries = auto_tag_topics(entries)

        report = AnalysisReport()
        report.total_feedback = len(entries)
        if not entries:
            return report

        # Global
        pos = sum(1 for e in entries if e.is_positive)
        report.positive_rate = pos / len(entries) * 100
        report.avg_quality = statistics.mean(e.quality_score for e in entries)

        # Topic analysis
        report.topics = self._analyze_topics(entries)
        report.worst_topics = [
            t.topic
            for t in sorted(report.topics, key=lambda t: t.failure_rate, reverse=True)[
                :3
            ]
            if t.failure_rate > 20
        ]
        report.best_topics = [
            t.topic
            for t in sorted(report.topics, key=lambda t: t.avg_quality, reverse=True)[
                :3
            ]
            if t.avg_quality >= 7
        ]

        # Tool analysis
        report.tools = self._analyze_tools(entries)
        report.worst_tools = [
            t.tool_name
            for t in sorted(report.tools, key=lambda t: t.failure_rate, reverse=True)[
                :3
            ]
            if t.failure_rate > 30
        ]

        # Time trend (weekly windows)
        report.time_trend = self._analyze_time_trend(entries)

        # Trending issues (topics getting worse recently)
        report.trending_issues = self._detect_trending_issues(entries)

        # Top corrections
        corrections = [e for e in entries if e.feedback_type == FeedbackType.CORRECTION]
        report.top_corrections = [
            {"query": e.query[:100], "correction": e.correction[:200], "topic": e.topic}
            for e in sorted(corrections, key=lambda e: -e.timestamp)[:10]
        ]

        return report

    # ── Topic breakdown ──────────────────────────────────────────────────

    def _analyze_topics(self, entries: list[FeedbackEntry]) -> list[TopicAnalysis]:
        by_topic: dict[str, list[FeedbackEntry]] = defaultdict(list)
        for e in entries:
            by_topic[e.topic or "general"].append(e)

        results = []
        for topic, group in by_topic.items():
            ta = TopicAnalysis(topic=topic)
            ta.total = len(group)
            ta.positive = sum(1 for e in group if e.is_positive)
            ta.negative = ta.total - ta.positive
            ta.avg_quality = statistics.mean(e.quality_score for e in group)
            ta.failure_rate = ta.negative / ta.total * 100
            ta.sample_queries = [e.query[:80] for e in group[:5]]
            ta.common_complaints = [
                e.comment[:100] for e in group if e.comment and not e.is_positive
            ][:5]
            results.append(ta)

        return sorted(results, key=lambda t: -t.total)

    # ── Tool breakdown ───────────────────────────────────────────────────

    def _analyze_tools(self, entries: list[FeedbackEntry]) -> list[ToolAnalysis]:
        by_tool: dict[str, list[FeedbackEntry]] = defaultdict(list)
        for e in entries:
            for tool in e.tools_used:
                by_tool[tool].append(e)

        results = []
        for tool_name, group in by_tool.items():
            ta = ToolAnalysis(tool_name=tool_name)
            ta.total_uses = len(group)
            ta.positive_when_used = sum(1 for e in group if e.is_positive)
            ta.negative_when_used = ta.total_uses - ta.positive_when_used
            ta.failure_rate = ta.negative_when_used / ta.total_uses * 100
            results.append(ta)

        return sorted(results, key=lambda t: -t.total_uses)

    # ── Time trend (weekly) ──────────────────────────────────────────────

    def _analyze_time_trend(self, entries: list[FeedbackEntry]) -> list[TimeWindow]:
        if not entries:
            return []

        WEEK = 7 * 24 * 3600
        oldest = min(e.timestamp for e in entries)
        newest = max(e.timestamp for e in entries)

        windows: list[TimeWindow] = []
        cursor = oldest
        week_num = 1
        while cursor <= newest + WEEK:
            end = cursor + WEEK
            bucket = [e for e in entries if cursor <= e.timestamp < end]
            if bucket:
                tw = TimeWindow(
                    label=f"Week {week_num}",
                    start=cursor,
                    end=end,
                    total=len(bucket),
                    positive=sum(1 for e in bucket if e.is_positive),
                    avg_quality=statistics.mean(e.quality_score for e in bucket),
                )
                windows.append(tw)
            cursor = end
            week_num += 1

        return windows

    # ── Trending issues ──────────────────────────────────────────────────

    def _detect_trending_issues(self, entries: list[FeedbackEntry]) -> list[str]:
        """Find topics where negative feedback is increasing recently."""
        if len(entries) < 4:
            return []

        # Split into first half vs second half
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        mid = len(sorted_entries) // 2
        first_half = sorted_entries[:mid]
        second_half = sorted_entries[mid:]

        def topic_fail_rate(group: list[FeedbackEntry]) -> dict[str, float]:
            by_topic: dict[str, list[FeedbackEntry]] = defaultdict(list)
            for e in group:
                by_topic[e.topic or "general"].append(e)
            return {
                topic: sum(1 for e in g if not e.is_positive) / max(len(g), 1) * 100
                for topic, g in by_topic.items()
            }

        early = topic_fail_rate(first_half)
        late = topic_fail_rate(second_half)

        issues = []
        for topic in late:
            early_rate = early.get(topic, 0)
            late_rate = late[topic]
            if late_rate > early_rate + 10 and late_rate > 20:
                issues.append(
                    f"{topic}: failure rate rose from {early_rate:.0f}% to {late_rate:.0f}%"
                )
        return sorted(issues)

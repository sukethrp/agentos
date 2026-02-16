"""Report — aggregate simulation results into a comprehensive summary.

Produces a :class:`SimulationReport` with:

* Global stats (total, pass rate, avg quality, avg latency)
* Per-persona breakdown (which moods the agent handles well / poorly)
* Failure analysis (common failure reasons, worst interactions)
* Time-series data for charting (quality over time, latency over time)
* Difficulty-vs-quality scatter data
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from agentos.simulation.evaluator import InteractionResult


@dataclass
class PersonaStats:
    """Aggregated stats for a single persona type."""

    name: str
    mood: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    avg_relevance: float = 0.0
    avg_helpfulness: float = 0.0
    avg_tone: float = 0.0
    avg_safety: float = 0.0
    total_cost: float = 0.0
    failure_reasons: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mood": self.mood,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 1),
            "avg_quality": round(self.avg_quality, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_relevance": round(self.avg_relevance, 2),
            "avg_helpfulness": round(self.avg_helpfulness, 2),
            "avg_tone": round(self.avg_tone, 2),
            "avg_safety": round(self.avg_safety, 2),
            "total_cost": round(self.total_cost, 5),
            "failure_reasons": self.failure_reasons[:5],
        }


@dataclass
class SimulationReport:
    """Full report from a simulation run."""

    # Global
    total_interactions: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_errors: int = 0
    pass_rate: float = 0.0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost: float = 0.0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    throughput_rps: float = 0.0

    # Breakdowns
    per_persona: list[PersonaStats] = field(default_factory=list)
    weakest_personas: list[str] = field(default_factory=list)
    strongest_personas: list[str] = field(default_factory=list)
    failure_reasons: dict[str, int] = field(default_factory=dict)

    # Chart data
    quality_over_time: list[dict] = field(default_factory=list)       # [{idx, overall}]
    latency_over_time: list[dict] = field(default_factory=list)       # [{idx, latency_ms}]
    persona_quality_chart: list[dict] = field(default_factory=list)   # [{persona, avg_quality}]
    difficulty_vs_quality: list[dict] = field(default_factory=list)   # [{difficulty, overall}]
    score_distribution: dict[str, int] = field(default_factory=dict)  # {"1-2": n, "3-4": n, ...}

    # Worst interactions (for debugging)
    worst_interactions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_interactions": self.total_interactions,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "total_errors": self.total_errors,
            "pass_rate": round(self.pass_rate, 1),
            "avg_quality": round(self.avg_quality, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_cost": round(self.total_cost, 5),
            "total_tokens": self.total_tokens,
            "duration_seconds": round(self.duration_seconds, 2),
            "throughput_rps": round(self.throughput_rps, 2),
            "per_persona": [p.to_dict() for p in self.per_persona],
            "weakest_personas": self.weakest_personas,
            "strongest_personas": self.strongest_personas,
            "failure_reasons": self.failure_reasons,
            "quality_over_time": self.quality_over_time,
            "latency_over_time": self.latency_over_time,
            "persona_quality_chart": self.persona_quality_chart,
            "difficulty_vs_quality": self.difficulty_vs_quality,
            "score_distribution": self.score_distribution,
            "worst_interactions": self.worst_interactions[:10],
        }

    def summary_text(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "  SIMULATION REPORT",
            "=" * 60,
            f"  Total interactions : {self.total_interactions}",
            f"  Passed             : {self.total_passed}  ({self.pass_rate:.1f}%)",
            f"  Failed             : {self.total_failed}",
            f"  Errors             : {self.total_errors}",
            f"  Avg quality        : {self.avg_quality:.2f} / 10",
            f"  Avg latency        : {self.avg_latency_ms:.0f} ms",
            f"  Total cost         : ${self.total_cost:.4f}",
            f"  Duration           : {self.duration_seconds:.1f}s",
            f"  Throughput         : {self.throughput_rps:.1f} req/s",
            "",
            "  Per-Persona Breakdown:",
        ]
        for p in self.per_persona:
            bar = "█" * int(p.pass_rate / 10)
            lines.append(
                f"    {p.name:<28} {p.pass_rate:5.1f}%  {bar:<10}  "
                f"quality={p.avg_quality:.1f}  n={p.total}"
            )

        if self.weakest_personas:
            lines.append(f"\n  ⚠️  Weakest: {', '.join(self.weakest_personas)}")
        if self.strongest_personas:
            lines.append(f"  ✅ Strongest: {', '.join(self.strongest_personas)}")

        if self.failure_reasons:
            lines.append("\n  Top Failure Reasons:")
            for reason, count in sorted(self.failure_reasons.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"    • {reason}: {count}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Builder ──────────────────────────────────────────────────────────────────

def build_report(
    results: list[InteractionResult],
    duration_seconds: float = 0.0,
    persona_difficulty: dict[str, float] | None = None,
) -> SimulationReport:
    """Build a full report from a list of scored interaction results."""
    rpt = SimulationReport()
    rpt.total_interactions = len(results)
    rpt.duration_seconds = duration_seconds
    rpt.throughput_rps = len(results) / max(duration_seconds, 0.01)

    if not results:
        return rpt

    # Global aggregates
    rpt.total_passed = sum(1 for r in results if r.passed)
    rpt.total_failed = sum(1 for r in results if not r.passed and not r.error)
    rpt.total_errors = sum(1 for r in results if r.error)
    rpt.pass_rate = rpt.total_passed / rpt.total_interactions * 100
    rpt.avg_quality = statistics.mean(r.overall for r in results)
    rpt.avg_latency_ms = statistics.mean(r.latency_ms for r in results)
    rpt.total_cost = sum(r.cost_usd for r in results)
    rpt.total_tokens = sum(r.tokens_used for r in results)

    # Per-persona
    by_persona: dict[str, list[InteractionResult]] = defaultdict(list)
    for r in results:
        by_persona[r.persona_name].append(r)

    persona_stats: list[PersonaStats] = []
    for name, group in by_persona.items():
        ps = PersonaStats(name=name, mood=group[0].persona_mood)
        ps.total = len(group)
        ps.passed = sum(1 for r in group if r.passed)
        ps.failed = ps.total - ps.passed
        ps.avg_quality = statistics.mean(r.overall for r in group)
        ps.avg_latency_ms = statistics.mean(r.latency_ms for r in group)
        ps.avg_relevance = statistics.mean(r.relevance for r in group)
        ps.avg_helpfulness = statistics.mean(r.helpfulness for r in group)
        ps.avg_tone = statistics.mean(r.tone for r in group)
        ps.avg_safety = statistics.mean(r.safety for r in group)
        ps.total_cost = sum(r.cost_usd for r in group)
        ps.failure_reasons = [r.failure_reason for r in group if r.failure_reason]
        persona_stats.append(ps)

    persona_stats.sort(key=lambda p: -p.avg_quality)
    rpt.per_persona = persona_stats

    if persona_stats:
        rpt.strongest_personas = [p.name for p in persona_stats[:2] if p.pass_rate >= 60]
        rpt.weakest_personas = [p.name for p in persona_stats[-2:] if p.pass_rate < 80]

    # Failure reasons
    reasons: Counter = Counter()
    for r in results:
        if r.failure_reason:
            reasons[r.failure_reason] += 1
    rpt.failure_reasons = dict(reasons.most_common(10))

    # Chart data
    rpt.quality_over_time = [{"idx": r.interaction_id, "overall": r.overall} for r in results]
    rpt.latency_over_time = [{"idx": r.interaction_id, "latency_ms": round(r.latency_ms, 1)} for r in results]
    rpt.persona_quality_chart = [
        {"persona": p.name, "mood": p.mood, "avg_quality": round(p.avg_quality, 2), "pass_rate": round(p.pass_rate, 1)}
        for p in persona_stats
    ]

    # Difficulty vs quality scatter
    diff_map = persona_difficulty or {}
    rpt.difficulty_vs_quality = [
        {"persona": r.persona_name, "difficulty": diff_map.get(r.persona_name, 0.5), "overall": r.overall}
        for r in results
    ]

    # Score distribution buckets
    buckets = {"1-2": 0, "3-4": 0, "5-6": 0, "7-8": 0, "9-10": 0}
    for r in results:
        s = r.overall
        if s <= 2:
            buckets["1-2"] += 1
        elif s <= 4:
            buckets["3-4"] += 1
        elif s <= 6:
            buckets["5-6"] += 1
        elif s <= 8:
            buckets["7-8"] += 1
        else:
            buckets["9-10"] += 1
    rpt.score_distribution = buckets

    # Worst interactions
    sorted_results = sorted(results, key=lambda r: r.overall)
    rpt.worst_interactions = [r.to_dict() for r in sorted_results[:10]]

    return rpt

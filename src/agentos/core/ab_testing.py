"""A/B testing utilities for AgentOS agents.

Uses the existing Simulation Sandbox (LLM-as-judge) to compare two agents
on the same set of queries.

Also includes rigorous statistical testing helpers:

1. Welch's t-test for unequal variances
2. Mann-Whitney U for non-parametric comparison
3. Bootstrap confidence intervals
4. Effect size (Cohen's d)
5. Sample size estimation
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from agentos.core.agent import Agent
from agentos.core.memory import Memory
from agentos.sandbox.scenario import Scenario
from agentos.sandbox.runner import Sandbox


@dataclass
class ABTestResult:
    """Results of an A/B test between two agent variants."""

    variant_a_name: str
    variant_b_name: str
    n_queries: int
    mean_a: float
    mean_b: float
    ci_a: Tuple[float, float]
    ci_b: Tuple[float, float]
    welch_t_statistic: float
    welch_p_value: float
    mann_whitney_u: float
    mann_whitney_p: float
    cohens_d: float
    effect_interpretation: str
    winner: Optional[str]
    confidence: float

    def summary(self) -> str:
        """Human-readable summary of the A/B test."""
        if self.winner:
            return (
                f"{self.winner} wins with {self.confidence:.1%} confidence "
                f"(Cohen's d = {self.cohens_d:.2f}, {self.effect_interpretation} effect). "
                f"Mean scores: {self.mean_a:.2f} vs {self.mean_b:.2f} "
                f"over {self.n_queries} queries."
            )
        return (
            f"No significant difference (p={self.welch_p_value:.3f}). "
            f"Mean scores: {self.mean_a:.2f} vs {self.mean_b:.2f}. "
            "Consider running more queries for statistical power."
        )


def welch_t_test(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    """Welch's t-test for unequal variances."""
    if len(scores_a) < 2 or len(scores_b) < 2:
        return 0.0, 1.0
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    n_a, n_b = len(a), len(b)
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0
    t_stat = (mean_a - mean_b) / se

    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    try:
        from scipy import stats

        p_value = 2 * stats.t.sf(abs(t_stat), df)
    except ImportError:
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    return float(t_stat), float(max(0.0, min(1.0, p_value)))


def cohens_d(scores_a: List[float], scores_b: List[float]) -> Tuple[float, str]:
    """Cohen's d effect size with interpretation."""
    if len(scores_a) < 2 or len(scores_b) < 2:
        return 0.0, "negligible"
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    n_a, n_b = len(a), len(b)

    pooled_std = np.sqrt(
        ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0, "negligible"

    d = abs(a.mean() - b.mean()) / pooled_std
    if d < 0.2:
        interpretation = "negligible"
    elif d < 0.5:
        interpretation = "small"
    elif d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    return float(d), interpretation


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not scores:
        return 0.0, 0.0
    arr = np.array(scores, dtype=float)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))

    alpha = 1 - ci
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def estimate_sample_size(effect_size: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Estimate per-variant sample size for two-sample comparisons."""
    effect_size = abs(effect_size)
    if effect_size <= 0:
        return 1000000
    try:
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
    except ImportError:
        z_alpha = 1.96
        z_beta = 0.84
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return max(2, int(math.ceil(n)))


def run_ab_test(
    scores_a: List[float],
    scores_b: List[float],
    name_a: str = "Variant A",
    name_b: str = "Variant B",
    significance_level: float = 0.05,
) -> ABTestResult:
    """Run a complete A/B statistical comparison between two score arrays."""
    if not scores_a or not scores_b:
        raise ValueError("scores_a and scores_b must not be empty")
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    t_stat, t_p = welch_t_test(scores_a, scores_b)

    def _norm_sf(x: float) -> float:
        # Survival function for the standard normal distribution.
        return 0.5 * math.erfc(x / math.sqrt(2.0))

    def _mann_whitney_two_sided_fallback(
        a: List[float], b: List[float]
    ) -> Tuple[float, float]:
        """Compute Mann–Whitney U and a two-sided p-value without SciPy.

        Uses rank-sum + normal approximation (with tie correction).
        """

        x = np.asarray(a, dtype=float)
        y = np.asarray(b, dtype=float)
        n1 = int(x.size)
        n2 = int(y.size)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0

        # Rank all values combined (average ranks for ties).
        combined = np.concatenate([x, y])
        order = np.argsort(combined, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)

        # Assign average ranks for tied groups.
        i = 0
        while i < combined.size:
            j = i + 1
            while j < combined.size and combined[order[j]] == combined[order[i]]:
                j += 1
            # ranks positions are 1-indexed
            avg_rank = 0.5 * (i + 1 + j)
            for k in range(i, j):
                ranks[order[k]] = avg_rank
            i = j

        # Rank sum for group A (mask first n1 elements which correspond to `a`).
        mask_a = np.zeros(combined.size, dtype=bool)
        mask_a[:n1] = True
        r1 = float(np.sum(ranks[mask_a]))

        # U statistic for A.
        u1 = n1 * n2 + n1 * (n1 + 1) / 2.0 - r1
        mean_u = n1 * n2 / 2.0
        n = n1 + n2

        # Tie correction: var(U) = n1*n2/12 * ((n+1) - tie_sum/(n*(n-1)))
        # where tie_sum = sum(t^3 - t) over tie groups.
        # Compute tie group sizes from the combined sorted values.
        tie_sum = 0
        sorted_vals = combined[order]
        i = 0
        while i < sorted_vals.size:
            j = i + 1
            while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
                j += 1
            t = j - i
            if t > 1:
                tie_sum += t**3 - t
            i = j

        if n <= 1:
            return float(u1), 1.0

        var_u = (n1 * n2 / 12.0) * (n + 1 - tie_sum / (n * (n - 1)))
        if var_u <= 0:
            return float(u1), 1.0

        # Continuity correction for normal approximation.
        z = (u1 - mean_u) / math.sqrt(var_u)
        # Two-sided p-value from |z|.
        p = 2.0 * _norm_sf(abs(z))
        p = max(0.0, min(1.0, float(p)))

        return float(u1), p

    try:
        from scipy.stats import mannwhitneyu

        u_stat, u_p = mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        u_stat = float(u_stat)
        u_p = float(u_p)
    except ImportError:
        u_stat, u_p = _mann_whitney_two_sided_fallback(scores_a, scores_b)

    d, d_interp = cohens_d(scores_a, scores_b)
    ci_a = bootstrap_ci(scores_a)
    ci_b = bootstrap_ci(scores_b)

    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))

    # Winner gate: use Mann–Whitney significance only. If significant, pick
    # the variant with the higher mean.
    winner: Optional[str] = None
    if u_p < significance_level:
        winner = name_a if mean_a > mean_b else name_b

    confidence = max(0.0, 1.0 - float(u_p))

    return ABTestResult(
        variant_a_name=name_a,
        variant_b_name=name_b,
        n_queries=len(scores_a),
        mean_a=mean_a,
        mean_b=mean_b,
        ci_a=ci_a,
        ci_b=ci_b,
        welch_t_statistic=t_stat,
        welch_p_value=t_p,
        mann_whitney_u=u_stat,
        mann_whitney_p=u_p,
        cohens_d=d,
        effect_interpretation=d_interp,
        winner=winner,
        confidence=confidence,
    )


def clone_agent(agent: Agent, new_name: str) -> Agent:
    """Create an exact copy of an agent with a different name.

    Copies:
      - config (model, system_prompt, temperature, max_iterations)
      - tools list
      - memory (deep copy, so A/B tests don't mutate original)
    """
    memory_copy: Memory = copy.deepcopy(agent.memory)
    return Agent(
        name=new_name,
        model=agent.config.model,
        tools=list(agent.tools),
        system_prompt=agent.config.system_prompt,
        max_iterations=agent.config.max_iterations,
        temperature=agent.config.temperature,
        memory=memory_copy,
    )


class ABTestPerQueryResult(BaseModel):
    """Per-query comparison between two agents."""

    query: str
    run_index: int = 1
    score_a: float
    score_b: float
    winner: str  # "agent_a" | "agent_b" | "tie"
    reasoning_a: str = ""
    reasoning_b: str = ""


class ABTestScores(BaseModel):
    """Aggregate scores for an agent in an A/B test."""

    avg_overall: float = 0.0
    pass_rate: float = 0.0
    avg_quality: float = 0.0
    avg_relevance: float = 0.0
    avg_safety: float = 0.0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    win_rate: float = 0.0


class ABTestReport(BaseModel):
    """Full A/B test report."""

    agent_a_name: str
    agent_b_name: str
    winner: str  # "agent_a" | "agent_b" | "tie"
    confidence: float = 0.0  # 0.0–1.0 based on simple binomial test
    scores: dict[str, ABTestScores] = Field(default_factory=dict)
    per_query: List[ABTestPerQueryResult] = Field(default_factory=list)

    def print_report(self) -> None:
        print(f"\n{'=' * 60}")
        print("🧪 AgentOS A/B Test Report")
        print(f"{'=' * 60}")
        print(f"   Agent A: {self.agent_a_name}")
        print(f"   Agent B: {self.agent_b_name}")
        print(f"   Winner:  {self.winner}  (confidence: {self.confidence * 100:.1f}%)")
        print(f"{'-' * 60}")

        a = self.scores.get("agent_a")
        b = self.scores.get("agent_b")
        if a and b:
            print(
                f"   Avg overall (A vs B): {a.avg_overall:.2f} vs {b.avg_overall:.2f}"
            )
            print(f"   Pass rate   (A vs B): {a.pass_rate:.1f}% vs {b.pass_rate:.1f}%")
            print(
                f"   Win rate    (A vs B): {a.win_rate * 100:.1f}% vs {b.win_rate * 100:.1f}%"
            )
            print(
                f"   Total cost  (A vs B): ${a.total_cost:.4f} vs ${b.total_cost:.4f}"
            )
            print(
                f"   Total time  (A vs B): {a.total_latency_ms:.0f}ms vs {b.total_latency_ms:.0f}ms"
            )

        print("\n   Per-query breakdown:")
        print(f"   {'─' * 56}")
        for r in self.per_query:
            icon = (
                "A" if r.winner == "agent_a" else "B" if r.winner == "agent_b" else "="
            )
            print(f"   [{icon}] Q{r.run_index}: {r.query[:60]}")
            print(f"      Scores → A: {r.score_a:.1f} | B: {r.score_b:.1f}")
        print(f"{'=' * 60}")


class ABTest:
    """Run an A/B test between two agents using the Sandbox judge."""

    def __init__(self, agent_a: Agent, agent_b: Agent, pass_threshold: float = 6.0):
        # Work on clones so we don't mutate the originals
        self.agent_a = clone_agent(agent_a, agent_a.config.name + "-A")
        self.agent_b = clone_agent(agent_b, agent_b.config.name + "-B")
        self.pass_threshold = pass_threshold

    def run_test(self, queries: List[str], num_runs: int = 10) -> ABTestReport:
        """Run the same queries on both agents and compare.

        Args:
            queries: List of user query strings.
            num_runs: How many times to repeat the full query set
                      (helps average out LLM randomness).
        """
        if not queries:
            raise ValueError("queries must not be empty")

        # Build scenarios
        scenarios: List[Scenario] = []
        for run_idx in range(num_runs):
            for i, q in enumerate(queries):
                scenarios.append(
                    Scenario(
                        name=f"Q{i + 1}-run{run_idx + 1}",
                        user_message=q,
                        expected_behavior="Provide a correct, concise, and safe answer to the user's question.",
                    )
                )

        # Run sandbox for each agent
        sandbox_a = Sandbox(self.agent_a, pass_threshold=self.pass_threshold)
        sandbox_b = Sandbox(self.agent_b, pass_threshold=self.pass_threshold)

        report_a = sandbox_a.run(scenarios)
        report_b = sandbox_b.run(scenarios)

        # Per-scenario mapping
        map_a = {r.scenario_name: r for r in report_a.results}
        map_b = {r.scenario_name: r for r in report_b.results}

        per_query: List[ABTestPerQueryResult] = []
        wins_a = wins_b = ties = 0
        overall_a: List[float] = []
        overall_b: List[float] = []

        for s in scenarios:
            ra = map_a.get(s.name)
            rb = map_b.get(s.name)
            if not ra or not rb:
                continue
            sa = ra.overall_score
            sb = rb.overall_score
            overall_a.append(sa)
            overall_b.append(sb)

            if sa > sb:
                winner = "agent_a"
                wins_a += 1
            elif sb > sa:
                winner = "agent_b"
                wins_b += 1
            else:
                winner = "tie"
                ties += 1

            # Extract run index from name if present (Qx-runY)
            run_index = 1
            if "-run" in s.name:
                try:
                    run_index = int(s.name.split("-run")[-1])
                except ValueError:
                    run_index = 1

            per_query.append(
                ABTestPerQueryResult(
                    query=s.user_message,
                    run_index=run_index,
                    score_a=sa,
                    score_b=sb,
                    winner=winner,
                    reasoning_a=ra.judge_reasoning,
                    reasoning_b=rb.judge_reasoning,
                )
            )

        # Aggregate scores
        avg_overall_a = sum(overall_a) / len(overall_a) if overall_a else 0.0
        avg_overall_b = sum(overall_b) / len(overall_b) if overall_b else 0.0

        total_duels = wins_a + wins_b
        win_rate_a = wins_a / total_duels if total_duels > 0 else 0.0
        win_rate_b = wins_b / total_duels if total_duels > 0 else 0.0

        scores = {
            "agent_a": ABTestScores(
                avg_overall=round(avg_overall_a, 2),
                pass_rate=report_a.pass_rate,
                avg_quality=report_a.avg_quality,
                avg_relevance=report_a.avg_relevance,
                avg_safety=report_a.avg_safety,
                total_cost=report_a.total_cost,
                total_latency_ms=report_a.total_latency_ms,
                win_rate=round(win_rate_a, 3),
            ),
            "agent_b": ABTestScores(
                avg_overall=round(avg_overall_b, 2),
                pass_rate=report_b.pass_rate,
                avg_quality=report_b.avg_quality,
                avg_relevance=report_b.avg_relevance,
                avg_safety=report_b.avg_safety,
                total_cost=report_b.total_cost,
                total_latency_ms=report_b.total_latency_ms,
                win_rate=round(win_rate_b, 3),
            ),
        }

        # Determine winner and a simple statistical confidence using a sign test
        if total_duels == 0 or wins_a == wins_b:
            winner_label = "tie"
            confidence = 0.0
        else:
            if wins_a > wins_b:
                winner_label = "agent_a"
                wins = wins_a
            else:
                winner_label = "agent_b"
                wins = wins_b

            # one-sided binomial p-value: P(X >= wins | n, p=0.5)
            def p_ge(k: int, n: int) -> float:
                return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)

            p_one = p_ge(wins, total_duels)
            p_two = min(1.0, 2 * p_one)  # two-sided
            confidence = max(0.0, 1.0 - p_two)

        return ABTestReport(
            agent_a_name=self.agent_a.config.name,
            agent_b_name=self.agent_b.config.name,
            winner=winner_label,
            confidence=round(confidence, 4),
            scores=scores,
            per_query=per_query,
        )

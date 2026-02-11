"""A/B testing utilities for AgentOS agents.

Uses the existing Simulation Sandbox (LLM-as-judge) to compare two agents
on the same set of queries.
"""

from __future__ import annotations

import copy
import math
from typing import List

from pydantic import BaseModel, Field

from agentos.core.agent import Agent
from agentos.core.memory import Memory
from agentos.sandbox.scenario import Scenario
from agentos.sandbox.runner import Sandbox


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
        print(f"\n{'='*60}")
        print(f"🧪 AgentOS A/B Test Report")
        print(f"{'='*60}")
        print(f"   Agent A: {self.agent_a_name}")
        print(f"   Agent B: {self.agent_b_name}")
        print(f"   Winner:  {self.winner}  (confidence: {self.confidence*100:.1f}%)")
        print(f"{'-'*60}")

        a = self.scores.get("agent_a")
        b = self.scores.get("agent_b")
        if a and b:
            print(f"   Avg overall (A vs B): {a.avg_overall:.2f} vs {b.avg_overall:.2f}")
            print(f"   Pass rate   (A vs B): {a.pass_rate:.1f}% vs {b.pass_rate:.1f}%")
            print(f"   Win rate    (A vs B): {a.win_rate*100:.1f}% vs {b.win_rate*100:.1f}%")
            print(f"   Total cost  (A vs B): ${a.total_cost:.4f} vs ${b.total_cost:.4f}")
            print(f"   Total time  (A vs B): {a.total_latency_ms:.0f}ms vs {b.total_latency_ms:.0f}ms")

        print(f"\n   Per-query breakdown:")
        print(f"   {'─'*56}")
        for r in self.per_query:
            icon = "A" if r.winner == "agent_a" else "B" if r.winner == "agent_b" else "="
            print(f"   [{icon}] Q{r.run_index}: {r.query[:60]}")
            print(f"      Scores → A: {r.score_a:.1f} | B: {r.score_b:.1f}")
        print(f"{'='*60}")


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
                        name=f"Q{i+1}-run{run_idx+1}",
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


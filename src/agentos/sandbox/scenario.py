"""LLM-as-judge testing framework — data models for sandbox scenarios.

This module defines the Pydantic models that describe test scenarios, their
results, and aggregated reports.  The actual execution and judging logic lives
in :mod:`agentos.sandbox.runner`; this module is intentionally limited to
**data** so the models can be serialised, stored, and shared without pulling
in heavy dependencies.

**Scoring rubric:**

Each scenario is evaluated on three 0-10 dimensions by a *separate* LLM
call (the "judge"):

- **Relevance** — did the response address the user's question?
- **Quality** — was it accurate, helpful, and well-formatted?
- **Safety** — did the agent avoid forbidden actions and behave
  appropriately?

The overall score is the simple mean of these three.  A scenario passes when
*all* of the following hold:

1. ``overall_score >= pass_threshold`` (default 6.0)
2. ``cost_usd <= scenario.max_cost``
3. ``latency_ms <= scenario.max_latency_ms``

**Why use a separate LLM call for judging?**

Having the *same* model self-assess its own output introduces systematic
bias — models tend to rate their own answers highly.  A dedicated judge call
with ``temperature=0`` and a structured rubric produces more consistent,
reproducible evaluations.  In production you can swap in a stronger model
(e.g. GPT-4o) as the judge while testing a cheaper model as the agent.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class Scenario(BaseModel):
    """A single test scenario to run against an agent.

    Describes the user message to send, what behaviour is expected, and
    optional constraints (forbidden actions, required tools, budgets).

    Example::

        Scenario(
            name="Tip calculator",
            user_message="What's 20% tip on $85?",
            expected_behavior="Returns 17.00",
            required_tools=["calculator"],
            max_cost=0.05,
        )
    """

    name: str
    user_message: str
    expected_behavior: str
    forbidden_actions: list[str] = Field(
        default_factory=list,
        description="Tool names or behaviours the agent must NOT exhibit.",
    )
    required_tools: list[str] = Field(
        default_factory=list,
        description="Tools the agent is expected to invoke.",
    )
    max_cost: float = Field(
        0.10, description="Maximum allowed USD cost for this scenario."
    )
    max_latency_ms: float = Field(
        30000, description="Maximum allowed wall-clock time in milliseconds."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Arbitrary labels for filtering (e.g. 'math', 'safety').",
    )


class ScenarioResult(BaseModel):
    """Outcome of executing one :class:`Scenario` against an agent.

    Contains the agent's raw response, the three judge scores, cost and
    latency measurements, and whether the scenario passed or failed.
    """

    scenario_name: str
    passed: bool
    agent_response: str
    relevance_score: float = 0.0
    safety_score: float = 0.0
    quality_score: float = 0.0
    overall_score: float = Field(
        0.0, description="Mean of relevance, quality, and safety scores."
    )
    bleu_score: float = 0.0
    rouge_l_score: float = 0.0
    embedding_similarity: float | None = None
    lexical_overlap: float | None = None
    llm_judge_score: float = 0.0
    safety_keyword_flag: float = 0.0
    tool_accuracy: float = 0.0
    conciseness: float = 0.0
    metrics_overall_score: float = 0.0
    tools_used: list[str] = Field(default_factory=list)
    tools_expected: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cost_ok: bool = True
    latency_ok: bool = True
    judge_reasoning: str = Field(
        "", description="Free-text explanation from the LLM judge."
    )
    error: str | None = None


class SandboxReport(BaseModel):
    """Aggregated report from running all scenarios in a sandbox session.

    Provides pass/fail counts, average scores per dimension, and total
    cost/latency.  Call :meth:`print_report` for a human-readable summary.

    Example::

        report = sandbox.run(scenarios)
        assert report.pass_rate >= 80.0, "Quality gate failed"
    """

    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    avg_quality: float = 0.0
    avg_relevance: float = 0.0
    avg_safety: float = 0.0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    results: list[ScenarioResult] = Field(default_factory=list)
    failed_scenarios: list[str] = Field(default_factory=list)

    def print_report(self) -> None:
        """Print a formatted summary to stdout for CLI consumption."""
        print(f"\n{'=' * 60}")
        print("AgentOS Simulation Sandbox Report")
        print(f"{'=' * 60}")
        print(f"   Scenarios:     {self.total_scenarios}")
        print(f"   Passed:      {self.passed}")
        print(f"   Failed:      {self.failed}")
        print(f"   Pass rate:     {self.pass_rate:.1f}%")
        print(f"   Avg quality:   {self.avg_quality:.1f}/10")
        print(f"   Avg relevance: {self.avg_relevance:.1f}/10")
        print(f"   Avg safety:    {self.avg_safety:.1f}/10")
        print(f"   Total cost:    ${self.total_cost:.4f}")
        print(f"   Total time:    {self.total_latency_ms:.0f}ms")
        print(f"{'=' * 60}")

        if self.failed_scenarios:
            print("\n   Failed scenarios:")
            for name in self.failed_scenarios:
                print(f"      - {name}")

        print("\n   Detailed Results:")
        print(f"   {'─' * 56}")
        for r in self.results:
            icon = "" if r.passed else ""
            print(f"   {icon} {r.scenario_name}")
            print(
                f"      Quality: {r.quality_score:.1f} | Relevance: {r.relevance_score:.1f} | Safety: {r.safety_score:.1f} | Cost: ${r.cost_usd:.4f}"
            )
            emb = (
                f"{r.embedding_similarity:.2f}"
                if r.embedding_similarity is not None
                else "n/a"
            )
            lex = (
                f"{r.lexical_overlap:.2f}" if r.lexical_overlap is not None else "n/a"
            )
            print(
                f"      BLEU: {r.bleu_score:.2f} | ROUGE-L: {r.rouge_l_score:.2f} | "
                f"Embed: {emb} | Lexical: {lex} | SafetyFlag: {r.safety_keyword_flag:.2f} | "
                f"ToolAcc: {r.tool_accuracy:.2f} | Concise: {r.conciseness:.2f} | "
                f"Metrics Overall: {r.metrics_overall_score:.2f}"
            )
            if not r.passed:
                print(f"      Reason: {r.judge_reasoning[:100]}")
            print(f"   {'─' * 56}")

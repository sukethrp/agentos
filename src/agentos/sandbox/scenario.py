from __future__ import annotations
from pydantic import BaseModel, Field


class Scenario(BaseModel):
    """A single test scenario for an agent."""

    name: str
    user_message: str
    expected_behavior: str
    forbidden_actions: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    max_cost: float = 0.10
    max_latency_ms: float = 30000
    tags: list[str] = Field(default_factory=list)


class ScenarioResult(BaseModel):
    """Result of running one scenario."""

    scenario_name: str
    passed: bool
    agent_response: str
    relevance_score: float = 0.0
    safety_score: float = 0.0
    quality_score: float = 0.0
    overall_score: float = 0.0
    tools_used: list[str] = Field(default_factory=list)
    tools_expected: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cost_ok: bool = True
    latency_ok: bool = True
    judge_reasoning: str = ""
    error: str | None = None


class SandboxReport(BaseModel):
    """Full report from running all scenarios."""

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

    def print_report(self):
        print(f"\n{'=' * 60}")
        print("🧪 AgentOS Simulation Sandbox Report")
        print(f"{'=' * 60}")
        print(f"   Scenarios:     {self.total_scenarios}")
        print(f"   ✅ Passed:      {self.passed}")
        print(f"   ❌ Failed:      {self.failed}")
        print(f"   Pass rate:     {self.pass_rate:.1f}%")
        print(f"   Avg quality:   {self.avg_quality:.1f}/10")
        print(f"   Avg relevance: {self.avg_relevance:.1f}/10")
        print(f"   Avg safety:    {self.avg_safety:.1f}/10")
        print(f"   Total cost:    ${self.total_cost:.4f}")
        print(f"   Total time:    {self.total_latency_ms:.0f}ms")
        print(f"{'=' * 60}")

        if self.failed_scenarios:
            print("\n   ❌ Failed scenarios:")
            for name in self.failed_scenarios:
                print(f"      - {name}")

        print("\n   Detailed Results:")
        print(f"   {'─' * 56}")
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            print(f"   {icon} {r.scenario_name}")
            print(
                f"      Quality: {r.quality_score:.1f} | Relevance: {r.relevance_score:.1f} | Safety: {r.safety_score:.1f} | Cost: ${r.cost_usd:.4f}"
            )
            if not r.passed:
                print(f"      Reason: {r.judge_reasoning[:100]}")
            print(f"   {'─' * 56}")

"""Agent Version Control — Track changes, rollback, compare versions.

Like Git, but for AI agents. Every time you change a prompt, tool, or config,
a new version is created. You can rollback, compare, and A/B test.

Usage:
    vc = AgentVersionControl("my-agent")
    vc.save_version(agent, tag="v1.0", notes="Initial release")
    vc.save_version(agent_v2, tag="v1.1", notes="Improved prompt")

    # Compare versions
    vc.compare("v1.0", "v1.1")

    # Rollback
    old_config = vc.get_version("v1.0")

    # List all versions
    vc.list_versions()
"""

from __future__ import annotations
import time
import json
from typing import Iterable

from pydantic import BaseModel, Field

from agentos.core.agent import Agent
from agentos.core.ab_testing import ABTest, ABTestReport
from agentos.tools import get_builtin_tools


class AgentVersion(BaseModel):
    tag: str
    timestamp: float = Field(default_factory=time.time)
    time_readable: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    )
    notes: str = ""
    config: dict = Field(default_factory=dict)
    system_prompt: str = ""
    model: str = ""
    tools: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_iterations: int = 10
    test_results: dict | None = None


class AgentVersionControl:
    """Version control system for agents."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.versions: dict[str, AgentVersion] = {}
        self.history: list[str] = []
        self.current_tag: str | None = None

    def save_version(
        self, agent, tag: str, notes: str = "", test_results: dict | None = None
    ):
        """Save current agent state as a version."""
        version = AgentVersion(
            tag=tag,
            notes=notes,
            config=agent.config.model_dump(),
            system_prompt=agent.config.system_prompt,
            model=agent.config.model,
            tools=[t.name for t in agent.tools],
            temperature=agent.config.temperature,
            max_iterations=agent.config.max_iterations,
            test_results=test_results,
        )
        self.versions[tag] = version
        self.history.append(tag)
        self.current_tag = tag
        print(f"💾 Saved version [{tag}]: {notes}")

    def get_version(self, tag: str) -> AgentVersion | None:
        return self.versions.get(tag)

    def list_versions(self):
        print(f"\n{'=' * 60}")
        print(f"📋 Version History: {self.agent_name}")
        print(f"{'=' * 60}")
        for tag in self.history:
            v = self.versions[tag]
            current = " ← current" if tag == self.current_tag else ""
            print(f"   [{v.tag}] {v.time_readable} — {v.notes}{current}")
            print(
                f"      Model: {v.model} | Tools: {', '.join(v.tools)} | Temp: {v.temperature}"
            )
            if v.test_results:
                print(
                    f"      Tests: {v.test_results.get('pass_rate', 'N/A')}% pass rate"
                )
            print()
        print(f"{'=' * 60}")

    def compare(self, tag_a: str, tag_b: str):
        """Compare two versions side by side."""
        a = self.versions.get(tag_a)
        b = self.versions.get(tag_b)
        if not a or not b:
            print(f"❌ Version not found: {tag_a if not a else tag_b}")
            return

        print(f"\n{'=' * 60}")
        print(f"🔄 Comparing: [{tag_a}] vs [{tag_b}]")
        print(f"{'=' * 60}")

        diffs = []
        if a.model != b.model:
            diffs.append(f"   Model:       {a.model} → {b.model}")
        if a.system_prompt != b.system_prompt:
            diffs.append(
                f"   Prompt:      Changed ({len(a.system_prompt)} → {len(b.system_prompt)} chars)"
            )
        if a.tools != b.tools:
            added = set(b.tools) - set(a.tools)
            removed = set(a.tools) - set(b.tools)
            if added:
                diffs.append(f"   Tools added: {', '.join(added)}")
            if removed:
                diffs.append(f"   Tools removed: {', '.join(removed)}")
        if a.temperature != b.temperature:
            diffs.append(f"   Temperature: {a.temperature} → {b.temperature}")
        if a.max_iterations != b.max_iterations:
            diffs.append(f"   Max iters:   {a.max_iterations} → {b.max_iterations}")

        if a.test_results and b.test_results:
            pr_a = a.test_results.get("pass_rate", 0)
            pr_b = b.test_results.get("pass_rate", 0)
            diff = pr_b - pr_a
            icon = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            diffs.append(f"   Pass rate:   {pr_a}% → {pr_b}% {icon} ({diff:+.1f}%)")

            qa = a.test_results.get("avg_quality", 0)
            qb = b.test_results.get("avg_quality", 0)
            qdiff = qb - qa
            icon = "📈" if qdiff > 0 else "📉" if qdiff < 0 else "➡️"
            diffs.append(f"   Avg quality: {qa} → {qb} {icon} ({qdiff:+.1f})")

        if diffs:
            print("\n   Changes:")
            for d in diffs:
                print(d)
        else:
            print("\n   No differences found.")
        print(f"{'=' * 60}")

    def rollback(self, tag: str) -> dict | None:
        """Get config from a previous version for rollback."""
        v = self.versions.get(tag)
        if not v:
            print(f"❌ Version not found: {tag}")
            return None
        self.current_tag = tag
        print(f"⏪ Rolled back to version [{tag}]")
        return v.config

    def ab_test(
        self,
        tag_a: str,
        tag_b: str,
        test_queries: Iterable[str],
        num_runs: int = 5,
    ) -> ABTestReport | None:
        """Run an A/B test between two saved versions using the Sandbox judge.

        Args:
            tag_a: Version tag for agent A.
            tag_b: Version tag for agent B.
            test_queries: Iterable of user queries to test.
            num_runs: How many times to repeat the full query set.
        """
        a = self.versions.get(tag_a)
        b = self.versions.get(tag_b)
        if not a or not b:
            print(f"❌ Version not found: {tag_a if not a else tag_b}")
            return None

        queries = [q for q in test_queries if q.strip()]
        if not queries:
            raise ValueError("test_queries must contain at least one non-empty query")

        available_tools = get_builtin_tools()

        def build_agent_from_version(v: AgentVersion, suffix: str) -> Agent:
            tools = [available_tools[t] for t in v.tools if t in available_tools]
            return Agent(
                name=f"{self.agent_name}-{v.tag}{suffix}",
                model=v.model,
                tools=tools,
                system_prompt=v.system_prompt or v.config.get("system_prompt", ""),
                max_iterations=v.max_iterations,
                temperature=v.temperature,
            )

        agent_a = build_agent_from_version(a, "-A")
        agent_b = build_agent_from_version(b, "-B")

        ab = ABTest(agent_a, agent_b)
        report = ab.run_test(queries, num_runs=num_runs)

        # Attach a brief summary of the A/B test to each version's test_results
        summary_a = {
            "opponent": tag_b,
            "role": "A",
            "winner": report.winner,
            "confidence": report.confidence,
            "avg_overall": report.scores["agent_a"].avg_overall,
            "win_rate": report.scores["agent_a"].win_rate,
        }
        summary_b = {
            "opponent": tag_a,
            "role": "B",
            "winner": report.winner,
            "confidence": report.confidence,
            "avg_overall": report.scores["agent_b"].avg_overall,
            "win_rate": report.scores["agent_b"].win_rate,
        }

        a.test_results = a.test_results or {}
        b.test_results = b.test_results or {}

        a.test_results.setdefault("ab_tests", {})[tag_b] = summary_a
        b.test_results.setdefault("ab_tests", {})[tag_a] = summary_b

        print(
            f"\n🧪 A/B test [{tag_a}] vs [{tag_b}] → winner: {report.winner} (conf {report.confidence * 100:.1f}%)"
        )

        return report

    def export_json(self) -> str:
        data = {tag: v.model_dump() for tag, v in self.versions.items()}
        return json.dumps(data, indent=2, default=str)

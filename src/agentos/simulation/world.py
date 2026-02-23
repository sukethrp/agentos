"""SimulatedWorld — the orchestrator for agent stress testing.

Creates a realistic business environment: picks personas, generates
traffic, dispatches queries concurrently, evaluates every response, and
produces a comprehensive report.

Usage::

    from agentos.core.agent import Agent
    from agentos.simulation import SimulatedWorld

    agent = Agent(name="support-bot")
    world = SimulatedWorld(agent)
    report = world.run(total=75, pattern="burst", concurrency=8)
    print(report.summary_text())
"""

from __future__ import annotations

import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from agentos.simulation.evaluator import Evaluator, InteractionResult
from agentos.simulation.personas import (
    ALL_PERSONAS,
    Persona,
    get_weighted_personas,
)
from agentos.simulation.report import SimulationReport, build_report
from agentos.simulation.traffic import (
    TrafficConfig,
    TrafficPattern,
    generate_traffic,
    describe_pattern,
)


@dataclass
class WorldConfig:
    """Tuneable knobs for the simulated world."""

    total_interactions: int = 50
    concurrency: int = 5
    traffic_pattern: TrafficPattern = TrafficPattern.STEADY
    requests_per_second: float = 2.0
    use_llm_judge: bool = False
    pass_threshold: float = 6.0
    personas: list[Persona] | None = None  # None → use weighted defaults
    quiet: bool = False  # suppress per-interaction prints


class SimulatedWorld:
    """Spin up a simulated business environment and stress-test an agent."""

    def __init__(
        self,
        agent: Any,  # Agent instance (or any object with .run(str)->Message)
        config: WorldConfig | None = None,
        on_interaction: Callable[[InteractionResult], None] | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or WorldConfig()
        self.evaluator = Evaluator(
            use_llm_judge=self.config.use_llm_judge,
            pass_threshold=self.config.pass_threshold,
        )
        self._results: list[InteractionResult] = []
        self._lock = threading.Lock()
        self._on_interaction = on_interaction
        self._running = False
        self._progress = 0

    # ── Public API ───────────────────────────────────────────────────────

    def run(
        self,
        total: int | None = None,
        pattern: str | None = None,
        concurrency: int | None = None,
    ) -> SimulationReport:
        """Execute the full simulation and return the report."""
        cfg = self.config
        if total is not None:
            cfg.total_interactions = total
        if pattern is not None:
            cfg.traffic_pattern = TrafficPattern(pattern)
        if concurrency is not None:
            cfg.concurrency = concurrency

        personas = cfg.personas or get_weighted_personas(cfg.total_interactions)
        traffic_cfg = TrafficConfig(
            pattern=cfg.traffic_pattern,
            total_requests=cfg.total_interactions,
            requests_per_second=cfg.requests_per_second,
        )

        if not cfg.quiet:
            print("\n🌐 Simulation World Starting")
            print(f"   Agent: {getattr(self.agent, 'config', {})}")
            print(f"   {describe_pattern(traffic_cfg)}")
            print(f"   Concurrency: {cfg.concurrency}")
            print(f"   LLM judge: {'yes' if cfg.use_llm_judge else 'no (heuristic)'}")
            print("-" * 60)

        self._results = []
        self._progress = 0
        self._running = True
        start = time.time()

        # Build the work queue: (delay, interaction_id, persona, query)
        work_items: list[tuple[float, int, Persona, str]] = []
        traffic = list(generate_traffic(traffic_cfg))
        for delay, idx in traffic:
            persona = personas[idx % len(personas)]
            query = persona.generate_query()
            work_items.append((delay, idx, persona, query))

        # Dispatch with concurrency limit
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futures = {}
            for delay, idx, persona, query in work_items:
                time.sleep(delay)
                fut = pool.submit(self._run_one, idx, persona, query)
                futures[fut] = idx

            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    if not cfg.quiet:
                        print(f"  ⚠️  Interaction error: {exc}")

        elapsed = time.time() - start
        self._running = False

        # Build difficulty map for the report
        diff_map = {p.name: p.difficulty for p in ALL_PERSONAS}

        report = build_report(
            self._results, duration_seconds=elapsed, persona_difficulty=diff_map
        )

        if not cfg.quiet:
            print(f"\n{report.summary_text()}")

        return report

    @property
    def progress(self) -> float:
        """Fraction completed (0-1)."""
        total = self.config.total_interactions
        return self._progress / max(total, 1)

    @property
    def results(self) -> list[InteractionResult]:
        with self._lock:
            return list(self._results)

    @property
    def running(self) -> bool:
        return self._running

    # ── Internal ─────────────────────────────────────────────────────────

    def _run_one(self, idx: int, persona: Persona, query: str) -> InteractionResult:
        """Execute a single interaction, evaluate, and store."""
        result = InteractionResult(
            interaction_id=idx,
            persona_name=persona.name,
            persona_mood=persona.mood.value,
            query=query,
            response="",
        )

        t0 = time.time()
        try:
            # Suppress agent prints during simulation
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                msg = self.agent.run(query)
                result.response = (
                    msg.content or "" if hasattr(msg, "content") else str(msg)
                )
                # Gather cost/token info from agent events
                if hasattr(self.agent, "events"):
                    for ev in self.agent.events:
                        result.tokens_used += getattr(ev, "tokens_used", 0)
                        result.cost_usd += getattr(ev, "cost_usd", 0.0)
            finally:
                sys.stdout = old_stdout
        except Exception as exc:
            result.error = str(exc)

        result.latency_ms = (time.time() - t0) * 1000

        # Evaluate
        self.evaluator.evaluate(result)

        with self._lock:
            self._results.append(result)
            self._progress += 1

        if self._on_interaction:
            try:
                self._on_interaction(result)
            except Exception:
                pass

        if not self.config.quiet:
            status = "✅" if result.passed else ("❌" if result.error else "⚠️")
            print(
                f"  {status} [{idx:3d}] {persona.name:<28} "
                f"quality={result.overall:.1f}  "
                f"{result.latency_ms:.0f}ms"
            )

        return result

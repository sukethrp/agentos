"""Self-optimization loop for AgentOS configs.

Orchestrates sandbox metrics, A/B testing, embedding drift detection,
governance checks, and observability tracing. Does not reimplement those
subsystems — it wires them together.

A dashboard panel for live optimization runs is a follow-up; this module
exposes the API and result objects only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agentos.core.ab_testing import ABTestResult, run_ab_test
from agentos.governance.guardrails import GovernanceEngine
from agentos.observability.tracer import TraceBuilder, TraceStore, get_trace_store
from agentos.rag.drift import DriftReport, EmbeddingDriftDetector
from agentos.sandbox.metrics import evaluate_response


@dataclass
class EvalExample:
    """One row in an offline evaluation set."""

    input: str
    expected: str
    expected_tools: list[str] | None = None


@dataclass
class RunOutput:
    """Structured output from a variant run function."""

    response: str
    tools_called: list[str] = field(default_factory=list)


RunFn = Callable[[Any, EvalExample], RunOutput | str]


@dataclass
class VariantStats:
    """Aggregated scores for one config variant."""

    name: str
    config: Any
    scores: list[float]
    mean_score: float
    trace_ids: list[str] = field(default_factory=list)


@dataclass
class ChallengerDecision:
    """Outcome of comparing the current champion to one challenger."""

    challenger_name: str
    adopted: bool
    ab_result: ABTestResult
    significance_test: str
    reason: str


@dataclass
class OptimizationResult:
    """Full result from a self-optimization pass."""

    chosen_config: Any
    chosen_name: str
    variant_stats: dict[str, VariantStats]
    decisions: list[ChallengerDecision]
    decision_rationale: str
    drift_detected: bool
    drift_report: DriftReport | None = None
    re_evaluation_rounds: int = 0


class SelfOptimizer:
    """Run eval, statistical comparison, drift checks, and adoption logic."""

    def __init__(
        self,
        alpha: float = 0.05,
        effect_floor: float = 0.2,
        drift_threshold: float = 0.1,
        cost_per_eval: float = 0.0,
        governance: GovernanceEngine | None = None,
        trace_store: TraceStore | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.alpha = alpha
        self.effect_floor = effect_floor
        self.drift_threshold = drift_threshold
        self.cost_per_eval = cost_per_eval
        self.governance = governance
        self.trace_store = trace_store or get_trace_store()
        self.embedder = embedder

    def optimize(
        self,
        eval_set: list[EvalExample],
        incumbent_config: Any,
        incumbent_name: str,
        candidates: list[tuple[str, Any]],
        run_fn: RunFn,
        baseline_embeddings: list[list[float]] | None = None,
    ) -> OptimizationResult:
        """Evaluate variants, compare with A/B tests, and pick a config."""
        self._check_governance("optimize:start")

        variant_stats = self._evaluate_all_variants(
            eval_set,
            [(incumbent_name, incumbent_config), *candidates],
            run_fn,
        )

        drift_detected = False
        drift_report: DriftReport | None = None
        re_evaluation_rounds = 0

        if baseline_embeddings is not None and self.embedder is not None:
            current_embeddings = self._collect_eval_embeddings(eval_set)
            drift_report, drift_detected = self._check_drift(
                baseline_embeddings, current_embeddings
            )
            if drift_detected:
                re_evaluation_rounds = 1
                variant_stats = self._evaluate_all_variants(
                    eval_set,
                    [(incumbent_name, incumbent_config), *candidates],
                    run_fn,
                )
                self._emit_decision_trace(
                    "drift_reevaluation",
                    {
                        "drift_detected": True,
                        "mmd_score": drift_report.mmd_score if drift_report else None,
                        "rounds": re_evaluation_rounds,
                    },
                )

        chosen_name = incumbent_name
        chosen_config = incumbent_config
        decisions: list[ChallengerDecision] = []

        champion_scores = variant_stats[incumbent_name].scores
        for challenger_name, challenger_config in candidates:
            self._check_governance(f"compare:{challenger_name}")
            challenger_scores = variant_stats[challenger_name].scores
            ab_result = run_ab_test(
                champion_scores,
                challenger_scores,
                name_a=chosen_name,
                name_b=challenger_name,
                significance_level=self.alpha,
            )
            adopted, reason = self._should_adopt(ab_result, challenger_name)
            decision = ChallengerDecision(
                challenger_name=challenger_name,
                adopted=adopted,
                ab_result=ab_result,
                significance_test="welch",
                reason=reason,
            )
            decisions.append(decision)
            self._emit_decision_trace(
                f"adoption:{challenger_name}",
                {
                    "adopted": adopted,
                    "reason": reason,
                    "welch_p": ab_result.welch_p_value,
                    "mann_whitney_p": ab_result.mann_whitney_p,
                    "cohens_d": ab_result.cohens_d,
                    "mean_champion": ab_result.mean_a,
                    "mean_challenger": ab_result.mean_b,
                    "ci_champion": ab_result.ci_a,
                    "ci_challenger": ab_result.ci_b,
                },
            )
            if adopted:
                chosen_name = challenger_name
                chosen_config = challenger_config
                champion_scores = challenger_scores

        rationale = self._build_rationale(chosen_name, incumbent_name, decisions)
        return OptimizationResult(
            chosen_config=chosen_config,
            chosen_name=chosen_name,
            variant_stats=variant_stats,
            decisions=decisions,
            decision_rationale=rationale,
            drift_detected=drift_detected,
            drift_report=drift_report,
            re_evaluation_rounds=re_evaluation_rounds,
        )

    def _evaluate_all_variants(
        self,
        eval_set: list[EvalExample],
        variants: list[tuple[str, Any]],
        run_fn: RunFn,
    ) -> dict[str, VariantStats]:
        stats: dict[str, VariantStats] = {}
        for name, config in variants:
            scores, trace_ids = self._evaluate_variant(eval_set, name, config, run_fn)
            stats[name] = VariantStats(
                name=name,
                config=config,
                scores=scores,
                mean_score=float(np.mean(scores)) if scores else 0.0,
                trace_ids=trace_ids,
            )
        return stats

    def _evaluate_variant(
        self,
        eval_set: list[EvalExample],
        variant_name: str,
        config: Any,
        run_fn: RunFn,
    ) -> tuple[list[float], list[str]]:
        self._check_governance(f"variant:{variant_name}")
        builder = TraceBuilder(agent_name=f"optimizer:{variant_name}")
        scores: list[float] = []
        trace_ids: list[str] = []

        for example in eval_set:
            self._check_governance(f"eval:{variant_name}")
            if self.governance is not None and self.cost_per_eval > 0:
                result = self.governance.check_tool_call(
                    "optimizer_eval", estimated_cost=self.cost_per_eval
                )
                if not result.allowed:
                    raise RuntimeError(
                        f"Governance blocked eval for {variant_name}: {result.message}"
                    )

            raw = run_fn(config, example)
            if isinstance(raw, str):
                output = RunOutput(response=raw)
            else:
                output = raw

            report = evaluate_response(
                response=output.response,
                expected=example.expected,
                tools_called=output.tools_called,
                expected_tools=example.expected_tools,
                embedder=self.embedder,
            )
            scores.append(report.overall_score)

            builder.set_query(example.input)
            builder.add_final_answer(output.response[:200])
            if self.governance is not None and self.cost_per_eval > 0:
                self.governance.record_action(
                    "optimizer_eval", cost=self.cost_per_eval, success=True
                )

        trace = builder.finish()
        self.trace_store.add(trace)
        trace_ids.append(trace.trace_id)
        self._emit_decision_trace(
            f"variant_run:{variant_name}",
            {"n_examples": len(eval_set), "mean_score": float(np.mean(scores))},
        )
        return scores, trace_ids

    def _collect_eval_embeddings(self, eval_set: list[EvalExample]) -> list[list[float]]:
        texts = [f"{ex.input}\n{ex.expected}" for ex in eval_set]
        return self.embedder.embed(texts)

    def _check_drift(
        self,
        baseline_embeddings: list[list[float]],
        current_embeddings: list[list[float]],
    ) -> tuple[DriftReport, bool]:
        detector = EmbeddingDriftDetector(threshold=self.drift_threshold)
        detector.set_reference(baseline_embeddings)
        report = detector.check(current_embeddings)
        self._emit_decision_trace(
            "drift_check",
            {
                "mmd_score": report.mmd_score,
                "threshold": report.threshold,
                "is_drifted": report.is_drifted,
            },
        )
        return report, report.is_drifted

    def _should_adopt(
        self, ab_result: ABTestResult, challenger_name: str
    ) -> tuple[bool, str]:
        challenger_wins = ab_result.mean_b > ab_result.mean_a
        significant = ab_result.welch_p_value < self.alpha
        effect_ok = ab_result.cohens_d >= self.effect_floor

        if challenger_wins and significant and effect_ok:
            return (
                True,
                (
                    f"Adopted {challenger_name}: Welch p={ab_result.welch_p_value:.4f} "
                    f"< {self.alpha}, Cohen's d={ab_result.cohens_d:.2f} "
                    f">= {self.effect_floor}, "
                    f"CI champion {ab_result.ci_a}, CI challenger {ab_result.ci_b}"
                ),
            )

        parts: list[str] = []
        if not challenger_wins:
            parts.append(
                f"challenger mean {ab_result.mean_b:.4f} "
                f"<= champion mean {ab_result.mean_a:.4f}"
            )
        if not significant:
            parts.append(
                f"Welch p={ab_result.welch_p_value:.4f} >= alpha {self.alpha}"
            )
        if not effect_ok:
            parts.append(
                f"Cohen's d={ab_result.cohens_d:.2f} < floor {self.effect_floor}"
            )
        return False, "; ".join(parts)

    def _check_governance(self, action: str) -> None:
        if self.governance is None:
            return
        if self.governance.killed:
            raise RuntimeError(
                f"Governance kill switch active: {self.governance.kill_reason}"
            )
        if self.cost_per_eval <= 0:
            return
        result = self.governance.check_tool_call(
            action, estimated_cost=self.cost_per_eval
        )
        if not result.allowed and result.rule == "kill_switch":
            raise RuntimeError(f"Governance blocked {action}: {result.message}")

    def _emit_decision_trace(self, event: str, payload: dict) -> None:
        builder = TraceBuilder(agent_name="optimizer")
        builder.set_query(event)
        builder.add_final_answer(str(payload))
        self.trace_store.add(builder.finish())

    def _build_rationale(
        self,
        chosen_name: str,
        incumbent_name: str,
        decisions: list[ChallengerDecision],
    ) -> str:
        if chosen_name == incumbent_name:
            if not decisions:
                return f"Kept incumbent {incumbent_name}; no challengers evaluated."
            last = decisions[-1]
            return (
                f"Kept incumbent {incumbent_name}. "
                f"Last challenger {last.challenger_name}: {last.reason}"
            )
        adopted = [d for d in decisions if d.adopted]
        final = adopted[-1]
        return (
            f"Adopted {chosen_name} over {incumbent_name}. "
            f"{final.significance_test} test: {final.reason}"
        )

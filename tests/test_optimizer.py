from __future__ import annotations

import numpy as np
import pytest

from agentos.governance.guardrails import GovernanceEngine
from agentos.optimize.optimizer import EvalExample, SelfOptimizer


def _eval_set(n: int, expected: str = "refund within five business days") -> list[EvalExample]:
    return [
        EvalExample(input=f"case-{i}", expected=expected) for i in range(n)
    ]


def test_adopts_clearly_better_variant() -> None:
    eval_set = _eval_set(60)
    incumbent = {"prompt": "vague"}
    challenger = {"prompt": "precise"}

    def run_fn(config: dict, example: EvalExample) -> str:
        if config["prompt"] == "precise":
            return example.expected
        return "contact support eventually"

    optimizer = SelfOptimizer(alpha=0.05, effect_floor=0.2)
    result = optimizer.optimize(
        eval_set=eval_set,
        incumbent_config=incumbent,
        incumbent_name="incumbent",
        candidates=[("challenger", challenger)],
        run_fn=run_fn,
    )

    assert result.chosen_name == "challenger"
    assert result.chosen_config == challenger
    assert result.decisions[0].adopted is True
    assert result.decisions[0].ab_result.welch_p_value < 0.05
    assert result.decisions[0].ab_result.cohens_d >= 0.2
    assert "Adopted challenger" in result.decision_rationale
    assert result.variant_stats["challenger"].mean_score > result.variant_stats["incumbent"].mean_score


def test_rejects_non_significant_improvement() -> None:
    expected = "reset password via settings security tab"
    eval_set = _eval_set(8, expected=expected)
    incumbent = {"variant": "a"}
    challenger = {"variant": "b"}

    def run_fn(config: dict, example: EvalExample) -> str:
        if config["variant"] == "b":
            return expected
        return expected.replace("security", "account")

    optimizer = SelfOptimizer(alpha=0.05, effect_floor=0.2)
    result = optimizer.optimize(
        eval_set=eval_set,
        incumbent_config=incumbent,
        incumbent_name="incumbent",
        candidates=[("challenger", challenger)],
        run_fn=run_fn,
    )

    assert result.chosen_name == "incumbent"
    assert result.decisions[0].adopted is False
    assert result.decisions[0].ab_result.welch_p_value >= 0.05
    assert "Kept incumbent" in result.decision_rationale


def test_drift_triggers_reevaluation() -> None:
    class StubEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [list(vec) for vec in np.random.normal(5.0, 0.1, size=(len(texts), 8))]

    np.random.seed(0)
    baseline = np.random.normal(0.0, 0.1, size=(40, 8)).tolist()
    eval_set = _eval_set(12)

    def run_fn(_config: dict, example: EvalExample) -> str:
        return example.expected

    optimizer = SelfOptimizer(
        drift_threshold=0.05,
        embedder=StubEmbedder(),
    )
    result = optimizer.optimize(
        eval_set=eval_set,
        incumbent_config={"prompt": "base"},
        incumbent_name="incumbent",
        candidates=[],
        run_fn=run_fn,
        baseline_embeddings=baseline,
    )

    assert result.drift_detected is True
    assert result.re_evaluation_rounds == 1
    assert result.drift_report is not None
    assert result.drift_report.is_drifted is True


def test_governance_kill_switch_blocks_run() -> None:
    gov = GovernanceEngine(agent_name="opt-bot")
    gov.kill("test stop")

    optimizer = SelfOptimizer(governance=gov)
    with pytest.raises(RuntimeError, match="kill switch"):
        optimizer.optimize(
            eval_set=_eval_set(2),
            incumbent_config={},
            incumbent_name="incumbent",
            candidates=[],
            run_fn=lambda _c, ex: ex.expected,
        )

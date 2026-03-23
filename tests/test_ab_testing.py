from __future__ import annotations

import numpy as np

from agentos.core.ab_testing import (
    bootstrap_ci,
    cohens_d,
    run_ab_test,
)


def test_identical_distributions_no_winner() -> None:
    scores = [7.0, 7.5, 8.0, 6.5, 7.2, 7.8, 6.9, 7.1, 7.4, 7.3]
    result = run_ab_test(scores, list(scores), name_a="A", name_b="B")
    assert result.winner is None
    assert result.welch_p_value > 0.05


def test_different_distributions_correct_winner() -> None:
    np.random.seed(123)
    scores_a = np.random.normal(loc=5.0, scale=0.6, size=80).tolist()
    scores_b = np.random.normal(loc=6.2, scale=0.6, size=80).tolist()
    result = run_ab_test(scores_a, scores_b, name_a="A", name_b="B")
    assert result.winner == "B"
    assert result.welch_p_value < 0.05


def test_cohens_d_interpretation_thresholds() -> None:
    np.random.seed(7)

    # negligible
    base = np.random.normal(loc=1.0, scale=1.0, size=400)
    negligible = np.random.normal(loc=1.05, scale=1.0, size=400)
    d1, i1 = cohens_d(base.tolist(), negligible.tolist())
    assert d1 < 0.2
    assert i1 == "negligible"

    # small
    small = np.random.normal(loc=1.35, scale=1.0, size=400)
    d2, i2 = cohens_d(base.tolist(), small.tolist())
    assert 0.2 <= d2 < 0.5
    assert i2 == "small"

    # medium
    medium = np.random.normal(loc=1.65, scale=1.0, size=400)
    d3, i3 = cohens_d(base.tolist(), medium.tolist())
    assert 0.5 <= d3 < 0.8
    assert i3 == "medium"

    # large
    large = np.random.normal(loc=2.2, scale=1.0, size=400)
    d4, i4 = cohens_d(base.tolist(), large.tolist())
    assert d4 >= 0.8
    assert i4 == "large"


def test_bootstrap_ci_contains_true_mean() -> None:
    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=1.0, size=200).tolist()
    true_mean = float(np.mean(data))
    lower, upper = bootstrap_ci(data, n_bootstrap=1000, ci=0.95)
    assert lower <= true_mean <= upper

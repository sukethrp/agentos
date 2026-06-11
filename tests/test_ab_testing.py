from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from agentos.core.ab_testing import (
    ABTestResult,
    bootstrap_ci,
    cohens_d,
    estimate_sample_size,
    run_ab_test,
    welch_t_test,
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


def test_welch_t_test_closed_form() -> None:
    scores_a = [2.0, 4.0]
    scores_b = [6.0, 8.0]
    t_stat, p_value = welch_t_test(scores_a, scores_b)

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    expected_t = (a.mean() - b.mean()) / se
    num = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2
    denom = (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (
        len(b) - 1
    )
    df = num / denom
    expected_p = 2 * stats.t.sf(abs(expected_t), df)

    assert t_stat == pytest.approx(expected_t)
    assert p_value == pytest.approx(expected_p, rel=1e-9)


def test_cohens_d_closed_form() -> None:
    scores_a = [2.0, 4.0]
    scores_b = [6.0, 8.0]
    d, interpretation = cohens_d(scores_a, scores_b)

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    pooled_std = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    expected_d = abs(a.mean() - b.mean()) / pooled_std

    assert d == pytest.approx(expected_d)
    assert interpretation == "large"


def test_mann_whitney_fallback_matches_scipy() -> None:
    scores_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    scores_b = [1.5, 2.5, 6.0, 7.0, 8.0]
    scipy_u, scipy_p = stats.mannwhitneyu(
        scores_a, scores_b, alternative="two-sided", method="asymptotic"
    )

    with patch("scipy.stats.mannwhitneyu", side_effect=ImportError):
        result = run_ab_test(scores_a, scores_b)

    assert result.mann_whitney_u == pytest.approx(float(scipy_u), rel=1e-6)
    assert result.mann_whitney_p == pytest.approx(float(scipy_p), rel=1e-6)


def test_mann_whitney_fallback_with_ties_matches_scipy() -> None:
    scores_a = [1.0, 2.0, 3.0, 4.0]
    scores_b = [1.0, 2.0, 5.0, 6.0]
    scipy_u, scipy_p = stats.mannwhitneyu(
        scores_a, scores_b, alternative="two-sided", method="asymptotic"
    )

    with patch("scipy.stats.mannwhitneyu", side_effect=ImportError):
        result = run_ab_test(scores_a, scores_b)

    assert result.mann_whitney_u == pytest.approx(float(scipy_u), rel=1e-6)
    assert result.mann_whitney_p == pytest.approx(float(scipy_p), rel=1e-6)


def test_estimate_sample_size_monotonic_in_effect_size() -> None:
    small_effect = estimate_sample_size(0.2)
    medium_effect = estimate_sample_size(0.5)
    large_effect = estimate_sample_size(0.8)
    assert small_effect > medium_effect > large_effect
    assert all(n >= 2 for n in (small_effect, medium_effect, large_effect))


def test_estimate_sample_size_zero_effect_returns_large_n() -> None:
    assert estimate_sample_size(0.0) == 1_000_000


def test_welch_t_test_insufficient_samples_returns_neutral() -> None:
    t_stat, p_value = welch_t_test([1.0], [2.0, 3.0])
    assert t_stat == 0.0
    assert p_value == 1.0


def test_cohens_d_zero_variance_returns_negligible() -> None:
    d, interpretation = cohens_d([5.0, 5.0, 5.0], [7.0, 7.0, 7.0])
    assert d == 0.0
    assert interpretation == "negligible"


def test_run_ab_test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        run_ab_test([1.0, 2.0], [1.0])


def test_ab_test_result_summary_with_and_without_winner() -> None:
    winner_result = ABTestResult(
        variant_a_name="A",
        variant_b_name="B",
        n_queries=10,
        mean_a=8.0,
        mean_b=6.0,
        ci_a=(7.0, 9.0),
        ci_b=(5.0, 7.0),
        welch_t_statistic=2.0,
        welch_p_value=0.04,
        mann_whitney_u=5.0,
        mann_whitney_p=0.03,
        cohens_d=0.9,
        effect_interpretation="large",
        winner="A",
        confidence=0.97,
    )
    assert "A wins" in winner_result.summary()
    assert "large effect" in winner_result.summary()

    tie_result = ABTestResult(
        variant_a_name="A",
        variant_b_name="B",
        n_queries=10,
        mean_a=7.0,
        mean_b=7.1,
        ci_a=(6.0, 8.0),
        ci_b=(6.0, 8.0),
        welch_t_statistic=0.1,
        welch_p_value=0.92,
        mann_whitney_u=50.0,
        mann_whitney_p=0.80,
        cohens_d=0.05,
        effect_interpretation="negligible",
        winner=None,
        confidence=0.20,
    )
    assert "No significant difference" in tie_result.summary()

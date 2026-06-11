from __future__ import annotations

import numpy as np
import pytest

from agentos.rag.drift import EmbeddingDriftDetector, compute_mmd


def test_same_distribution_not_drifted() -> None:
    np.random.seed(1)
    reference = np.random.normal(0.0, 1.0, size=(200, 64))
    current = np.random.normal(0.0, 1.0, size=(200, 64))

    detector = EmbeddingDriftDetector(threshold=0.2)
    detector.set_reference(reference.tolist())
    report = detector.check(current.tolist())

    assert report.is_drifted is False
    assert report.mmd_score <= report.threshold


def test_shifted_distribution_detected_as_drifted() -> None:
    np.random.seed(2)
    reference = np.random.normal(0.0, 1.0, size=(200, 64))
    current = np.random.normal(1.0, 1.0, size=(200, 64))

    detector = EmbeddingDriftDetector(threshold=0.2)
    detector.set_reference(reference.tolist())
    report = detector.check(current.tolist())

    assert report.is_drifted is True
    assert report.mmd_score > report.threshold


def test_single_embedding_edge_case() -> None:
    detector = EmbeddingDriftDetector(threshold=0.05)
    detector.set_reference([[0.1, 0.2, 0.3]])
    report = detector.check([[0.11, 0.19, 0.31]])

    assert report.n_reference == 1
    assert report.n_current == 1
    assert report.mmd_score >= 0.0


def test_compute_mmd_identical_distributions_near_zero() -> None:
    np.random.seed(0)
    samples = np.random.normal(size=(50, 8))
    assert compute_mmd(samples, samples, kernel="rbf") < 0.05
    assert compute_mmd(samples, samples, kernel="linear") < 1e-9


def test_compute_mmd_shifted_distribution_grows() -> None:
    np.random.seed(1)
    reference = np.random.normal(loc=0.0, scale=1.0, size=(80, 16))
    shifted = np.random.normal(loc=2.0, scale=1.0, size=(80, 16))
    same = np.random.normal(loc=0.0, scale=1.0, size=(80, 16))

    mmd_same = compute_mmd(reference, same, kernel="rbf")
    mmd_shifted = compute_mmd(reference, shifted, kernel="rbf")
    assert mmd_shifted > mmd_same


def test_compute_mmd_symmetric() -> None:
    np.random.seed(3)
    x = np.random.normal(size=(30, 4))
    y = np.random.normal(loc=0.5, size=(25, 4))
    assert compute_mmd(x, y, kernel="rbf") == pytest.approx(
        compute_mmd(y, x, kernel="rbf"), rel=1e-9
    )
    assert compute_mmd(x, y, kernel="linear") == pytest.approx(
        compute_mmd(y, x, kernel="linear"), rel=1e-9
    )


def test_compute_mmd_unbiased_rbf_known_small_input() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    gamma = 1.0

    def rbf_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq_dists = (
            np.sum(a**2, axis=1, keepdims=True)
            + np.sum(b**2, axis=1, keepdims=True).T
            - 2 * a @ b.T
        )
        return np.exp(-gamma * sq_dists)

    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)
    n, m = len(x), len(y)
    expected_sq = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2 * k_xy.sum() / (n * m)
    )
    expected = float(max(0.0, expected_sq) ** 0.5)
    assert compute_mmd(x, y, kernel="rbf", gamma=gamma) == pytest.approx(expected)


def test_compute_mmd_linear_closed_form() -> None:
    x = np.array([[0.0, 0.0], [2.0, 0.0]])
    y = np.array([[10.0, 0.0], [12.0, 0.0]])
    assert compute_mmd(x, y, kernel="linear") == pytest.approx(10.0)


def test_compute_mmd_dimension_mismatch_raises() -> None:
    x = np.array([[1.0, 2.0]])
    y = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="same embedding dimension"):
        compute_mmd(x, y)


def test_compute_mmd_non_2d_raises() -> None:
    with pytest.raises(ValueError, match="2D arrays"):
        compute_mmd(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))


def test_compute_mmd_unknown_kernel_raises() -> None:
    x = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="kernel must be"):
        compute_mmd(x, x, kernel="poly")


def test_compute_mmd_empty_input_returns_zero() -> None:
    x = np.empty((0, 3))
    y = np.array([[1.0, 2.0, 3.0]])
    assert compute_mmd(x, y) == 0.0
    assert compute_mmd(y, x) == 0.0


def test_detector_rejects_empty_reference() -> None:
    detector = EmbeddingDriftDetector()
    with pytest.raises(ValueError, match="non-empty 2D"):
        detector.set_reference([])


def test_detector_rejects_empty_current() -> None:
    detector = EmbeddingDriftDetector()
    detector.set_reference([[1.0, 2.0]])
    with pytest.raises(ValueError, match="non-empty 2D"):
        detector.check([])


def test_detector_dimension_mismatch_raises() -> None:
    detector = EmbeddingDriftDetector()
    detector.set_reference([[1.0, 2.0]])
    with pytest.raises(ValueError, match="dimension must match"):
        detector.check([[1.0, 2.0, 3.0]])


def test_detector_requires_reference_before_check() -> None:
    detector = EmbeddingDriftDetector()
    with pytest.raises(RuntimeError, match="set_reference"):
        detector.check([[1.0, 2.0]])

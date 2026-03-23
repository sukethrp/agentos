from __future__ import annotations

import numpy as np

from agentos.rag.drift import EmbeddingDriftDetector


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

"""Embedding Drift Detection for the RAG pipeline.

In production, document content changes over time and embedding distributions
can shift, degrading retrieval quality. This module detects when the embedding
space has drifted significantly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DriftReport:
    """Report from an embedding drift check."""

    mmd_score: float
    threshold: float
    is_drifted: bool
    n_reference: int
    n_current: int
    mean_cosine_shift: float
    recommendation: str


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> float:
    """Compute Maximum Mean Discrepancy between two sets of embeddings."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same embedding dimension")
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    if kernel not in {"rbf", "linear"}:
        raise ValueError("kernel must be one of: 'rbf', 'linear'")

    if kernel == "linear":
        mean_diff = X.mean(axis=0) - Y.mean(axis=0)
        return float(np.sqrt(np.maximum(0.0, np.dot(mean_diff, mean_diff))))

    if gamma is None:
        try:
            from scipy.spatial.distance import cdist

            dists = cdist(X[:100], Y[:100], "sqeuclidean")
            gamma = 1.0 / (2 * max(float(np.median(dists)), 1e-10))
        except Exception:
            gamma = 1.0 / X.shape[1]

    def rbf_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq_dists = (
            np.sum(A**2, axis=1, keepdims=True)
            + np.sum(B**2, axis=1, keepdims=True).T
            - 2 * A @ B.T
        )
        return np.exp(-gamma * sq_dists)

    K_xx = rbf_kernel(X, X)
    K_yy = rbf_kernel(Y, Y)
    K_xy = rbf_kernel(X, Y)

    n, m = len(X), len(Y)
    # Handle tiny samples with a stable fallback.
    if n < 2 or m < 2:
        mmd_sq = max(0.0, float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()))
        return float(np.sqrt(mmd_sq))

    mmd_sq = (
        (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
        + (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
        - 2 * K_xy.sum() / (n * m)
    )
    return float(max(0.0, mmd_sq) ** 0.5)


class EmbeddingDriftDetector:
    """Monitor embedding distributions for drift over time."""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self._reference: Optional[np.ndarray] = None
        self._ref_mean: Optional[np.ndarray] = None

    def set_reference(self, embeddings: List[List[float]]) -> None:
        """Set the reference embedding distribution."""
        reference = np.array(embeddings, dtype=float)
        if reference.ndim != 2 or len(reference) == 0:
            raise ValueError("embeddings must be a non-empty 2D list")
        self._reference = reference
        self._ref_mean = reference.mean(axis=0)

    def check(self, current_embeddings: List[List[float]]) -> DriftReport:
        """Check current embeddings against reference for drift."""
        if self._reference is None or self._ref_mean is None:
            raise RuntimeError("Call set_reference() first")
        current = np.array(current_embeddings, dtype=float)
        if current.ndim != 2 or len(current) == 0:
            raise ValueError("current_embeddings must be a non-empty 2D list")
        if current.shape[1] != self._reference.shape[1]:
            raise ValueError("current embedding dimension must match reference")

        cur_mean = current.mean(axis=0)
        mmd = compute_mmd(self._reference, current)

        cos_sim = float(
            np.dot(self._ref_mean, cur_mean)
            / (np.linalg.norm(self._ref_mean) * np.linalg.norm(cur_mean) + 1e-10)
        )
        mean_shift = float(1 - cos_sim)
        is_drifted = mmd > self.threshold

        if is_drifted:
            rec = (
                "Embedding distribution has drifted significantly. "
                "Recommend re-indexing documents with fresh embeddings."
            )
        else:
            rec = "No significant drift detected. No action needed."

        return DriftReport(
            mmd_score=mmd,
            threshold=self.threshold,
            is_drifted=is_drifted,
            n_reference=len(self._reference),
            n_current=len(current),
            mean_cosine_shift=mean_shift,
            recommendation=rec,
        )

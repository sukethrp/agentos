from __future__ import annotations

import pytest

from agentos.sandbox.metrics import (
    MetricsReport,
    bleu_score,
    rouge_l_score,
    toxicity_score,
)


def test_bleu_known_pairs() -> None:
    ref = "the cat is on the mat"
    identical = "the cat is on the mat"
    unrelated = "completely different words here"

    assert bleu_score(ref, identical) > 0.99
    assert bleu_score(ref, unrelated) < 0.5


def test_rouge_l_known_lcs_cases() -> None:
    ref = "a b c d e"
    cand = "a x c y e"
    score = rouge_l_score(ref, cand)
    assert 0.59 <= score <= 0.61


def test_toxicity_low_for_safe_text() -> None:
    text = "I can help you summarize this document safely."
    assert toxicity_score(text) < 0.6


def test_toxicity_high_for_harmful_text() -> None:
    text = "Let's hack and exploit this target."
    assert toxicity_score(text) >= 0.8


def test_metrics_report_weighted_score_computation() -> None:
    report = MetricsReport(
        bleu_score=0.8,
        rouge_l_score=0.7,
        semantic_similarity=0.9,
        llm_judge_score=8.0,
        toxicity_score=0.1,
        tool_accuracy=1.0,
        conciseness=0.9,
    )
    expected = (
        0.10 * 0.8
        + 0.10 * 0.7
        + 0.25 * 0.9
        + 0.15 * (8.0 / 10.0)
        + 0.20 * (1 - 0.1)
        + 0.15 * 1.0
        + 0.05 * 0.9
    )
    assert report.overall_score == pytest.approx(expected)

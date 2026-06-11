from __future__ import annotations

import pytest

from agentos.sandbox.metrics import (
    MetricsReport,
    bleu_score,
    embedding_similarity,
    lexical_overlap,
    rouge_l_score,
    safety_keyword_flag,
)


class _FixedEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [1.0, 0.0]] if texts[0] == texts[1] else [[1.0, 0.0], [0.0, 1.0]]


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


def test_safety_keyword_flag_low_for_benign_text() -> None:
    text = "I can help you summarize this document safely."
    assert safety_keyword_flag(text) < 0.1
    assert safety_keyword_flag("") == 0.0


def test_safety_keyword_flag_high_for_harmful_text() -> None:
    text = "Let's hack and exploit this target."
    assert safety_keyword_flag(text) >= 0.4


def test_lexical_overlap_identical_strings() -> None:
    text = "the cat is on the mat"
    assert lexical_overlap(text, text) == 1.0


def test_embedding_similarity_identical_strings() -> None:
    text = "the cat is on the mat"
    assert embedding_similarity(text, text, _FixedEmbedder()) > 0.99


def test_metrics_report_weighted_score_computation() -> None:
    report = MetricsReport(
        bleu_score=0.8,
        rouge_l_score=0.7,
        embedding_similarity=0.9,
        lexical_overlap=None,
        llm_judge_score=8.0,
        safety_keyword_flag=0.1,
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


def test_metrics_report_lexical_fallback_in_overall_score() -> None:
    report = MetricsReport(
        bleu_score=0.0,
        rouge_l_score=0.0,
        embedding_similarity=None,
        lexical_overlap=0.8,
        llm_judge_score=0.0,
        safety_keyword_flag=0.0,
        tool_accuracy=0.0,
        conciseness=0.0,
    )
    assert report.overall_score == pytest.approx(0.25 * 0.8 + 0.20)

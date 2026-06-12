from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentos.rag.embeddings import (
    EmbeddingEngine,
    LocalEmbeddings,
    OpenAIEmbeddings,
    TFIDFEmbeddings,
    get_embeddings,
)


def test_get_embeddings_returns_tfidf_backend() -> None:
    pytest.importorskip("sklearn")
    backend = get_embeddings("tfidf", n_components=64)
    assert isinstance(backend, TFIDFEmbeddings)
    assert backend.dimension() == 64


@patch("agentos.rag.embeddings.OpenAI")
def test_get_embeddings_returns_openai_backend(mock_openai_cls: MagicMock) -> None:
    mock_openai_cls.return_value = MagicMock()
    backend = get_embeddings("openai")
    assert isinstance(backend, OpenAIEmbeddings)


def test_get_embeddings_raises_on_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        get_embeddings("bogus")


def test_tfidf_embeddings_unit_norm_and_dimension() -> None:
    pytest.importorskip("sklearn")
    corpus = [
        "the cat sat on the mat",
        "dogs run in the park",
        "birds fly above the trees",
        "fish swim in the river",
        "horses gallop across open fields",
        "rabbits hop through the garden",
    ]
    embedder = TFIDFEmbeddings(n_components=4)
    embedder.fit(corpus)
    vectors = embedder.embed(["the cat sat on the mat", "dogs run in the park"])

    assert embedder.dimension() == 4
    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 4
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_embedding_engine_cache_hit_and_disk_persistence(tmp_path) -> None:
    pytest.importorskip("sklearn")
    cache_dir = tmp_path / "cache"
    engine = EmbeddingEngine(backend="tfidf", n_components=4, cache_dir=str(cache_dir))

    warmup = [
        "persistent cache entry for embeddings",
        "second sentence expands the vocabulary",
        "third line adds more distinct terms",
        "fourth document ensures enough features",
    ]
    engine.embed_batch(warmup)

    text = warmup[0]
    first = engine.embed(text)
    second = engine.embed(text)

    assert first == second
    cache_file = cache_dir / "embeddings_cache.json"
    assert cache_file.exists()

    reloaded = json.loads(cache_file.read_text())
    key = engine._cache_key(text)
    assert key in reloaded
    assert reloaded[key] == first

    engine2 = EmbeddingEngine(backend="tfidf", n_components=4, cache_dir=str(cache_dir))
    assert engine2.embed(text) == first


@patch("agentos.rag.embeddings.OpenAI")
def test_openai_embeddings_batching(mock_openai_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    embedder = OpenAIEmbeddings()
    result = embedder.embed(["hello"])

    assert result == [[0.1, 0.2, 0.3]]
    assert embedder.dimension() == 1536
    mock_client.embeddings.create.assert_called_once()


def test_local_embeddings_dimension() -> None:
    pytest.importorskip("sentence_transformers")
    embedder = LocalEmbeddings()
    vectors = embedder.embed(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert embedder.dimension() == len(vectors[0])
    for vec in vectors:
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_tfidf_empty_inputs() -> None:
    pytest.importorskip("sklearn")
    embedder = TFIDFEmbeddings(n_components=4)
    assert embedder.embed([]) == []
    embedder.fit([])
    assert embedder._fitted is False


def test_embedding_engine_empty_batch() -> None:
    pytest.importorskip("sklearn")
    engine = EmbeddingEngine(backend="tfidf", n_components=4)
    assert engine.embed_batch([]) == []


def test_embedding_engine_recovers_from_corrupt_disk_cache(tmp_path) -> None:
    pytest.importorskip("sklearn")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "embeddings_cache.json").write_text("{not-json")
    engine = EmbeddingEngine(backend="tfidf", n_components=4, cache_dir=str(cache_dir))
    assert engine._cache == {}


@patch.dict("os.environ", {}, clear=True)
@patch("agentos.rag.embeddings.LocalEmbeddings", side_effect=ImportError)
def test_get_embeddings_auto_falls_back_to_tfidf(
    _mock_local: MagicMock,
) -> None:
    pytest.importorskip("sklearn")
    backend = get_embeddings("auto", n_components=4)
    assert isinstance(backend, TFIDFEmbeddings)

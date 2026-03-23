"""Embedding backends for the AgentOS RAG pipeline.

Supports multiple embedding strategies:
1. OpenAIEmbeddings - calls text-embedding-3-small (default, easy)
2. LocalEmbeddings - uses sentence-transformers locally (no API key)
3. TFIDFEmbeddings - lightweight TF-IDF baseline (zero API dependencies)

Design decision (ADR-007): We support local embeddings because production
deployments often cannot send data to external APIs for privacy/compliance.
Sentence Transformers models like all-MiniLM-L6-v2 provide strong quality at
zero API cost and low latency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
OPENAI_DIMENSIONS = 1536
MAX_BATCH_SIZE = 2048


class BaseEmbeddings(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""

    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimensionality."""


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI text-embedding-3-small embeddings."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model
        self._client = OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out

    def dimension(self) -> int:
        return OPENAI_DIMENSIONS


class LocalEmbeddings(BaseEmbeddings):
    """Local embeddings with sentence-transformers."""

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def dimension(self) -> int:
        return self._dimension


class TFIDFEmbeddings(BaseEmbeddings):
    """TF-IDF + TruncatedSVD baseline backend."""

    def __init__(self, max_features: int = 10000, n_components: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._svd = TruncatedSVD(n_components=n_components)
        self._fitted = False
        self._n_components = n_components

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            return
        tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self._svd.fit(tfidf_matrix)
        self._fitted = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._fitted:
            self.fit(texts)
        tfidf = self._vectorizer.transform(texts)
        reduced = self._svd.transform(tfidf)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = reduced / norms
        return normalized.tolist()

    def dimension(self) -> int:
        return self._n_components


def get_embeddings(backend: str = "auto", **kwargs: Any) -> BaseEmbeddings:
    """Factory that creates embedding backends.

    Backends: auto, openai, local, tfidf.
    """
    backend_key = backend.lower()
    if backend_key == "openai":
        return OpenAIEmbeddings(model=kwargs.get("model", DEFAULT_MODEL))
    if backend_key == "local":
        return LocalEmbeddings(model=kwargs.get("model", DEFAULT_LOCAL_MODEL))
    if backend_key == "tfidf":
        return TFIDFEmbeddings(
            max_features=kwargs.get("max_features", 10000),
            n_components=kwargs.get("n_components", 256),
        )
    if backend_key != "auto":
        raise ValueError(
            f"Unknown backend: {backend}. Choose from: ['auto', 'openai', 'local', 'tfidf']"
        )

    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model=kwargs.get("model", DEFAULT_MODEL))
    try:
        return LocalEmbeddings(model=kwargs.get("model", DEFAULT_LOCAL_MODEL))
    except ImportError:
        return TFIDFEmbeddings(
            max_features=kwargs.get("max_features", 10000),
            n_components=kwargs.get("n_components", 256),
        )


class EmbeddingEngine:
    """Compatibility wrapper around the new BaseEmbeddings backends."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        backend: str = "auto",
        **kwargs: Any,
    ):
        self.model = model
        self.backend = backend
        self._backend = get_embeddings(backend=backend, model=model, **kwargs)
        self._cache: dict[str, list[float]] = {}
        self._cache_path: Path | None = None

        if cache_dir:
            self._cache_path = Path(cache_dir)
            self._cache_path.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_fetch.append((i, text))

        if to_fetch:
            fetched = self._backend.embed([t for _, t in to_fetch])
            for (idx, text), vec in zip(to_fetch, fetched):
                key = self._cache_key(text)
                self._cache[key] = vec
                results[idx] = vec
            if self._cache_path:
                self._save_disk_cache()

        return results  # type: ignore[return-value]

    @property
    def dimensions(self) -> int:
        return self._backend.dimension()

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _load_disk_cache(self) -> None:
        if self._cache_path is None:
            return
        cache_file = self._cache_path / "embeddings_cache.json"
        if cache_file.exists():
            try:
                self._cache = json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save_disk_cache(self) -> None:
        if self._cache_path is None:
            return
        cache_file = self._cache_path / "embeddings_cache.json"
        cache_file.write_text(json.dumps(self._cache))

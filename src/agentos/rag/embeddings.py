"""Embedding generation using OpenAI text-embedding-3-small."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
MAX_BATCH_SIZE = 2048


class EmbeddingEngine:
    """Generate embeddings via OpenAI with optional disk cache."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
    ):
        self.model = model
        self.client = OpenAI()
        self._cache: dict[str, list[float]] = {}
        self._cache_path: Path | None = None

        if cache_dir:
            self._cache_path = Path(cache_dir)
            self._cache_path.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    # ── Public API ──

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Uses cache for previously seen inputs."""
        results: list[list[float] | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_fetch.append((i, text))

        if to_fetch:
            # Process in batches to respect API limits
            for batch_start in range(0, len(to_fetch), MAX_BATCH_SIZE):
                batch = to_fetch[batch_start : batch_start + MAX_BATCH_SIZE]
                batch_texts = [t for _, t in batch]

                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                )

                for j, embedding_obj in enumerate(response.data):
                    idx = batch[j][0]
                    text = batch[j][1]
                    vec = embedding_obj.embedding
                    key = self._cache_key(text)
                    self._cache[key] = vec
                    results[idx] = vec

            if self._cache_path:
                self._save_disk_cache()

        return results  # type: ignore[return-value]

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding model."""
        return DIMENSIONS

    # ── Cache helpers ──

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

"""In-memory vector store with cosine similarity search."""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from agentos.rag.base_store import BaseVectorStore
from agentos.rag.types import SearchResult


@dataclass
class Document:
    """A chunk of text with its embedding and metadata."""

    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


class VectorStore(BaseVectorStore):
    """Simple in-memory vector store using cosine similarity."""

    def __init__(self):
        self._documents: list[Document] = []

    # ── CRUD ──

    def add(self, text: str, embedding: list[float], metadata: dict | None = None, doc_id: str = "") -> int:
        """Add a single document. Returns its index."""
        doc = Document(
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            doc_id=doc_id,
        )
        self._documents.append(doc)
        return len(self._documents) - 1

    def add_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[int]:
        """Add multiple documents at once. Returns their indices."""
        metadatas = metadatas or [{} for _ in texts]
        doc_ids = doc_ids or ["" for _ in texts]
        indices = []
        for text, emb, meta, did in zip(texts, embeddings, metadatas, doc_ids):
            indices.append(self.add(text, emb, meta, did))
        return indices

    def search(self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        """Find the top-K most similar documents by cosine similarity."""
        if not self._documents:
            return []

        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._documents):
            sim = _cosine_similarity(query_embedding, doc.embedding)
            if sim >= threshold:
                scored.append((sim, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        return [
            SearchResult(
                text=self._documents[idx].text,
                score=score,
                metadata=self._documents[idx].metadata,
                doc_id=self._documents[idx].doc_id,
                index=idx,
            )
            for score, idx in top
        ]

    def delete(self, index: int) -> None:
        """Remove a document by index."""
        if 0 <= index < len(self._documents):
            self._documents.pop(index)

    def clear(self) -> None:
        """Remove all documents."""
        self._documents.clear()

    @property
    def size(self) -> int:
        return len(self._documents)

    # ── Persistence ──

    def save(self, path: str) -> None:
        """Save the store to a JSON file."""
        data = [
            {
                "text": d.text,
                "embedding": d.embedding,
                "metadata": d.metadata,
                "doc_id": d.doc_id,
            }
            for d in self._documents
        ]
        Path(path).write_text(json.dumps(data))

    def load(self, path: str) -> None:
        """Load documents from a JSON file (appends to existing)."""
        raw = json.loads(Path(path).read_text())
        for item in raw:
            self.add(
                text=item["text"],
                embedding=item["embedding"],
                metadata=item.get("metadata", {}),
                doc_id=item.get("doc_id", ""),
            )


# ── Math ──

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

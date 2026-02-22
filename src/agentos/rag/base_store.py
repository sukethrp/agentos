from __future__ import annotations
from abc import ABC, abstractmethod
from agentos.rag.types import SearchResult


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, text: str, embedding: list[float], metadata: dict | None = None, doc_id: str = "") -> int:
        pass

    @abstractmethod
    def add_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[int]:
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

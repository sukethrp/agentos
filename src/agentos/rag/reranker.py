from __future__ import annotations
from agentos.rag.types import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)
        scored = list(
            zip(results, scores.tolist() if hasattr(scores, "tolist") else list(scores))
        )
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            scored = scored[:top_k]
        return [
            SearchResult(
                text=r.text,
                score=float(s),
                metadata=r.metadata,
                doc_id=r.doc_id,
                index=r.index,
            )
            for r, s in scored
        ]

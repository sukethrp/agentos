from __future__ import annotations
import asyncio
import re

from agentos.rag.base_store import BaseVectorStore
from agentos.rag.embeddings import EmbeddingEngine
from agentos.rag.types import SearchResult


def _rrf_fuse(
    rankings: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank_list in rankings:
        for rank, (doc_id, _) in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in sorted_docs]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    def __init__(
        self,
        store: BaseVectorStore,
        embedder: EmbeddingEngine,
        corpus: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ):
        self._store: BaseVectorStore = store
        self._embedder = embedder
        self._corpus = corpus or []
        self._doc_ids = doc_ids or [f"doc_{i}" for i in range(len(self._corpus))]
        self._bm25 = None
        if self._corpus:
            from rank_bm25 import BM25Okapi

            tokenized = [_tokenize(t) for t in self._corpus]
            self._bm25 = BM25Okapi(tokenized)

    def set_corpus(self, corpus: list[str], doc_ids: list[str] | None = None) -> None:
        from rank_bm25 import BM25Okapi

        self._corpus = corpus
        self._doc_ids = doc_ids or [f"doc_{i}" for i in range(len(corpus))]
        tokenized = [_tokenize(t) for t in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.retrieve_async(query, top_k))

    async def retrieve_async(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_emb = self._embedder.embed(query)
        dense_task = asyncio.to_thread(
            self._store.search,
            query_emb,
            top_k=top_k * 2,
            threshold=0.0,
        )

        async def _bm25_async():
            if self._bm25:
                return await asyncio.to_thread(self._bm25_search, query, top_k * 2)
            return []

        dense_results, bm25_results = await asyncio.gather(dense_task, _bm25_async())
        dense_ranking = [(r.doc_id or str(r.index), r.score) for r in dense_results]
        bm25_ranking = bm25_results
        rankings = [r for r in [dense_ranking, bm25_ranking] if r]
        if not rankings:
            return []
        fused = _rrf_fuse(rankings)[:top_k]
        doc_map: dict[str, SearchResult] = {}
        for r in dense_results:
            key = r.doc_id or str(r.index)
            doc_map[key] = r
        for i, (doc_id, _) in enumerate(bm25_ranking):
            if doc_id not in doc_map and doc_id in self._doc_ids:
                idx = self._doc_ids.index(doc_id)
                doc_map[doc_id] = SearchResult(
                    text=self._corpus[idx],
                    score=0.0,
                    metadata={},
                    doc_id=doc_id,
                    index=idx,
                )
        results = []
        for doc_id, score in fused:
            if doc_id in doc_map:
                r = doc_map[doc_id]
                results.append(
                    SearchResult(
                        text=r.text,
                        score=score,
                        metadata=r.metadata,
                        doc_id=r.doc_id,
                        index=r.index,
                    )
                )
        return results[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [
            (self._doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0
        ]

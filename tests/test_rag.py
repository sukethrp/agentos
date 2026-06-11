from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from agentos.rag.chunker import Chunk, DocumentChunker
from agentos.rag.pipeline import RAGPipeline, RetrievalResult
from agentos.rag.retriever import HybridRetriever, _rrf_fuse
from agentos.rag.types import SearchResult
from agentos.rag.vector_store import VectorStore


class StubEmbeddingEngine:
    def __init__(self, dim: int = 4):
        self._dim = dim
        self._vocab = {
            "password": [1.0, 0.0, 0.0, 0.0],
            "reset": [0.9, 0.1, 0.0, 0.0],
            "billing": [0.0, 1.0, 0.0, 0.0],
            "invoice": [0.0, 0.9, 0.1, 0.0],
            "default": [0.1, 0.1, 0.1, 0.1],
        }

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            lowered = text.lower()
            vec = list(self._vocab["default"])
            for key, embedding in self._vocab.items():
                if key != "default" and key in lowered:
                    vec = embedding
                    break
            out.append(vec)
        return out

    @property
    def dimensions(self) -> int:
        return self._dim


class TestDocumentChunker:
    def test_empty_text_returns_no_chunks(self):
        chunker = DocumentChunker(min_chunk_size=10)
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   \n\n  ") == []

    def test_short_text_below_min_chunk_size_is_filtered(self):
        chunker = DocumentChunker(min_chunk_size=50)
        chunks = chunker.chunk_text("too short")
        assert chunks == []

    def test_paragraph_boundaries_are_preserved(self):
        text = "Alpha paragraph here.\n\nBeta paragraph follows."
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=10)
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert "Alpha paragraph" in chunks[0].text
        assert "Beta paragraph" in chunks[0].text

    def test_oversized_paragraph_splits_by_characters(self):
        para = "x" * 300
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        chunks = chunker.chunk_text(para)
        assert len(chunks) >= 3
        assert all(len(c.text) <= 100 for c in chunks)

    def test_character_split_overlap_is_shared_between_chunks(self):
        text = "segment " * 80
        chunker = DocumentChunker(chunk_size=60, chunk_overlap=15, min_chunk_size=10)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        assert chunks[0].text[-15:] in chunks[1].text

    def test_chunk_file_reads_markdown(self, tmp_path: Path):
        path = tmp_path / "notes.md"
        path.write_text("# Title\n\nBody text for chunking test here.")
        chunker = DocumentChunker(min_chunk_size=5)
        chunks = chunker.chunk_file(str(path))
        assert len(chunks) == 1
        assert chunks[0].metadata["filename"] == "notes.md"
        assert "Body text" in chunks[0].text


class TestRRFFuse:
    def test_fuses_rankings_prefers_consensus_top_doc(self):
        dense = [("doc_a", 0.9), ("doc_b", 0.8)]
        sparse = [("doc_b", 1.5), ("doc_c", 1.0)]
        fused = _rrf_fuse([dense, sparse])
        assert fused[0][0] == "doc_b"


class TestHybridRetriever:
    def test_ranks_password_doc_above_billing_for_password_query(self):
        corpus = [
            "To reset your password, open settings and choose security.",
            "Monthly billing invoices are available in the account portal.",
        ]
        doc_ids = ["password_doc", "billing_doc"]
        store = VectorStore()
        embedder = StubEmbeddingEngine()
        for i, text in enumerate(corpus):
            store.add(text, embedder.embed(text), {"filename": doc_ids[i]}, doc_ids[i])

        retriever = HybridRetriever(store, embedder, corpus=corpus, doc_ids=doc_ids)
        results = asyncio.run(
            retriever.retrieve_async("How do I reset my password?", top_k=2)
        )

        assert len(results) == 2
        assert results[0].doc_id == "password_doc"
        assert "password" in results[0].text.lower()

    def test_set_corpus_rebuilds_bm25_index(self):
        store = VectorStore()
        embedder = StubEmbeddingEngine()
        retriever = HybridRetriever(store, embedder)
        corpus = [
            "alpha beta gamma documentation",
            "unrelated zeta omega content",
        ]
        doc_ids = ["doc_alpha", "doc_other"]
        retriever.set_corpus(corpus, doc_ids)
        assert retriever._bm25 is not None
        assert retriever._corpus == corpus
        assert retriever._doc_ids == doc_ids


class TestRAGPipeline:
    def _pipeline_with_stub(self, **kwargs) -> RAGPipeline:
        with patch("agentos.rag.pipeline.EmbeddingEngine", return_value=StubEmbeddingEngine()):
            pipeline = RAGPipeline(**kwargs)
        pipeline.embedder = StubEmbeddingEngine()
        return pipeline

    def test_ingest_and_query_end_to_end_with_stub_embedder(self, tmp_path: Path):
        doc = tmp_path / "help.md"
        doc.write_text(
            "Password reset: open account settings, choose security, and click reset.\n\n"
            "Billing questions are handled by the finance team."
        )

        pipeline = self._pipeline_with_stub(top_k=2, similarity_threshold=0.0)

        added = pipeline.ingest(str(doc))
        assert added >= 1
        assert pipeline.num_chunks == added

        result = pipeline.query("How do I reset my password?")
        assert isinstance(result, RetrievalResult)
        assert result.top_text
        assert "password" in result.top_text.lower()
        assert "reset" in result.context.lower()
        assert result.results[0].metadata["filename"] == "help.md"

    def test_ingest_text_and_query_inline_source(self):
        pipeline = self._pipeline_with_stub(top_k=1)
        count = pipeline.ingest_text(
            "Invoice downloads are in the billing section of your dashboard.",
            source="billing-faq",
        )
        assert count >= 1
        result = pipeline.query("Where are invoices?")
        assert result.results
        assert "billing" in result.results[0].text.lower()

    def test_query_with_no_documents_returns_empty_context(self):
        pipeline = self._pipeline_with_stub()
        result = pipeline.query("anything")
        assert result.results == []
        assert result.context == "(No relevant documents found.)"

    def test_save_and_load_round_trip(self, tmp_path: Path):
        pipeline = self._pipeline_with_stub()
        pipeline.ingest_text("Persist me.", source="persist")
        store_path = tmp_path / "store.json"
        pipeline.save(str(store_path))

        loaded = self._pipeline_with_stub()
        loaded.load(str(store_path))
        assert loaded.num_chunks == pipeline.num_chunks

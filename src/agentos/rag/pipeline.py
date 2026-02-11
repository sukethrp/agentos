"""RAG Pipeline — ingest → chunk → embed → store → retrieve."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from agentos.rag.chunker import DocumentChunker, Chunk
from agentos.rag.embeddings import EmbeddingEngine
from agentos.rag.vector_store import VectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Result from a RAG query."""

    query: str
    results: list[SearchResult]
    context: str  # formatted text ready to inject into a prompt

    @property
    def top_text(self) -> str:
        """The highest-scoring chunk's text, or empty string."""
        return self.results[0].text if self.results else ""


class RAGPipeline:
    """End-to-end Retrieval Augmented Generation pipeline.

    Usage:
        rag = RAGPipeline()
        rag.ingest("docs/manual.pdf")
        rag.ingest("notes.md")
        result = rag.query("How do I reset my password?")
        print(result.context)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "text-embedding-3-small",
        cache_dir: str | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ):
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedder = EmbeddingEngine(
            model=embedding_model,
            cache_dir=cache_dir,
        )
        self.store = VectorStore()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self._ingested_files: list[str] = []

    # ── Ingest ──

    def ingest(self, path: str) -> int:
        """Ingest a single file. Returns the number of chunks added."""
        chunks = self.chunker.chunk_file(path)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        metadatas = [c.metadata for c in chunks]
        doc_ids = [f"{Path(path).name}:{c.index}" for c in chunks]

        self.store.add_batch(texts, embeddings, metadatas, doc_ids)
        self._ingested_files.append(path)

        return len(chunks)

    def ingest_text(self, text: str, source: str = "inline") -> int:
        """Ingest raw text directly. Returns the number of chunks added."""
        meta = {"source": source, "filename": source, "filetype": "text"}
        chunks = self.chunker.chunk_text(text, meta)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        metadatas = [c.metadata for c in chunks]
        doc_ids = [f"{source}:{c.index}" for c in chunks]

        self.store.add_batch(texts, embeddings, metadatas, doc_ids)
        return len(chunks)

    def ingest_directory(self, directory: str, extensions: list[str] | None = None) -> int:
        """Ingest all matching files in a directory. Returns total chunks added."""
        extensions = extensions or [".txt", ".md", ".pdf", ".rst"]
        total = 0
        for p in sorted(Path(directory).rglob("*")):
            if p.is_file() and p.suffix.lower() in extensions:
                total += self.ingest(str(p))
        return total

    # ── Query ──

    def query(self, question: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve the most relevant chunks for a question."""
        k = top_k or self.top_k
        query_embedding = self.embedder.embed(question)
        results = self.store.search(
            query_embedding,
            top_k=k,
            threshold=self.similarity_threshold,
        )
        context = self._format_context(results)
        return RetrievalResult(query=question, results=results, context=context)

    # ── Helpers ──

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results into a context block for the LLM."""
        if not results:
            return "(No relevant documents found.)"

        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("filename", r.doc_id or "unknown")
            parts.append(
                f"[Source {i}: {source} (score: {r.score:.3f})]\n{r.text}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Info ──

    @property
    def num_chunks(self) -> int:
        return self.store.size

    @property
    def ingested_files(self) -> list[str]:
        return list(self._ingested_files)

    def save(self, path: str) -> None:
        """Persist the vector store to disk."""
        self.store.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved vector store."""
        self.store.load(path)

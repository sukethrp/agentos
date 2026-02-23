from __future__ import annotations
import re
from pathlib import Path
from agentos.rag.chunker import DocumentChunker, Chunk
from agentos.rag.embeddings import EmbeddingEngine
from agentos.rag.base_store import BaseVectorStore
from agentos.rag.config import rag_config


def _chunk_fixed(
    text: str, chunk_size: int, chunk_overlap: int, metadata: dict
) -> list[Chunk]:
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if len(chunk_text) >= 50:
            chunks.append(
                Chunk(text=chunk_text, metadata={**metadata, "chunk_index": i}, index=i)
            )
            i += 1
        start += chunk_size - chunk_overlap
    return chunks


def _chunk_sentence(
    text: str, chunk_size: int, chunk_overlap: int, metadata: dict
) -> list[Chunk]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_len = 0
    i = 0
    for sent in sentences:
        sent_len = len(sent) + 1
        if current_len + sent_len > chunk_size and current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) >= 50:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={**metadata, "chunk_index": i},
                        index=i,
                    )
                )
                i += 1
            overlap = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 > chunk_overlap:
                    break
                overlap.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len
    if current:
        chunk_text = " ".join(current).strip()
        if len(chunk_text) >= 50:
            chunks.append(
                Chunk(text=chunk_text, metadata={**metadata, "chunk_index": i}, index=i)
            )
    return chunks


def _chunk_semantic(
    text: str, chunk_size: int, chunk_overlap: int, metadata: dict
) -> list[Chunk]:
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text, metadata)


CHUNK_STRATEGIES = {
    "fixed": _chunk_fixed,
    "sentence": _chunk_sentence,
    "semantic": _chunk_semantic,
}


_store_registry: dict[str, BaseVectorStore] = {}


def _get_store(collection_name: str) -> BaseVectorStore:
    if collection_name in _store_registry:
        return _store_registry[collection_name]
    from agentos.rag.vector_store import VectorStore

    vs = rag_config.get("vector_store", "chroma")
    if vs == "chroma":
        try:
            from agentos.rag.stores import ChromaStore

            store = ChromaStore(collection_name=collection_name)
        except ImportError:
            store = VectorStore()
    elif vs == "pinecone":
        from agentos.rag.stores import PineconeStore

        store = PineconeStore(index_name=collection_name)
    elif vs == "pgvector":
        from agentos.rag.stores import PgVectorStore

        store = PgVectorStore(collection_name=collection_name)
    else:
        store = VectorStore()
    _store_registry[collection_name] = store
    return store


class IngestionPipeline:
    def __init__(
        self,
        collection_name: str = "default",
        chunk_strategy: str = "fixed",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "text-embedding-3-small",
        store: BaseVectorStore | None = None,
    ):
        self.collection_name = collection_name
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._store = store or _get_store(collection_name)
        _store_registry[collection_name] = self._store
        self._embedder = EmbeddingEngine(model=embedding_model)
        self._chunk_fn = CHUNK_STRATEGIES.get(chunk_strategy, _chunk_fixed)

    def _read_file(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(str(path))
                return "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                import fitz

                doc = fitz.open(str(path))
                out = "\n\n".join(page.get_text() for page in doc)
                doc.close()
                return out
        return path.read_text(encoding="utf-8")

    def ingest_path(self, path: str) -> int:
        p = Path(path)
        if p.is_file():
            return self._ingest_file(p)
        total = 0
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in (
                ".txt",
                ".md",
                ".rst",
                ".csv",
                ".log",
                ".pdf",
            ):
                total += self._ingest_file(f)
        return total

    def _ingest_file(self, path: Path) -> int:
        text = self._read_file(path)
        meta = {
            "source": str(path),
            "filename": path.name,
            "filetype": path.suffix.lower(),
        }
        chunks = self._chunk_fn(text, self.chunk_size, self.chunk_overlap, meta)
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_batch(texts)
        metadatas = [c.metadata for c in chunks]
        doc_ids = [f"{path.name}:{c.index}" for c in chunks]
        self._store.add_batch(texts, embeddings, metadatas, doc_ids)
        return len(chunks)

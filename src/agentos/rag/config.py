from __future__ import annotations

rag_config: dict = {
    "vector_store": "chroma",
    "chunk_strategy": "fixed",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "embedding_backend": "auto",
    "embedding_model": "text-embedding-3-small",
}

from __future__ import annotations
import json
from agentos.core.tool import Tool, tool
from agentos.rag.ingestion import _get_store
from agentos.rag.embeddings import EmbeddingEngine
from agentos.rag.config import rag_config


def _rag_search_impl(query: str, collection: str, top_k: int) -> list[str]:
    store = _get_store(collection)
    embedder = EmbeddingEngine()
    query_emb = embedder.embed(query)
    results = store.search(query_emb, top_k=top_k, threshold=0.0)
    return [r.text for r in results]


@tool(
    name="rag_search",
    description="Search ingested documents in a RAG collection. Returns relevant text passages.",
)
def rag_search(query: str, collection: str = "default", top_k: int = 5) -> str:
    try:
        docs = _rag_search_impl(query, collection, top_k)
        return json.dumps(docs) if docs else "[]"
    except Exception as e:
        return json.dumps([f"ERROR: {e}"])


def rag_search_tool() -> Tool:
    return rag_search

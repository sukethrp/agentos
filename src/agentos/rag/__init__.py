"""AgentOS RAG — Retrieval Augmented Generation pipeline.

Usage:
    from agentos.rag import RAGPipeline

    rag = RAGPipeline()
    rag.ingest("docs/manual.pdf")
    rag.ingest("notes.md")

    result = rag.query("How do I reset my password?")
    print(result.context)

As an agent tool:
    from agentos.tools.rag_tool import create_rag_tool
    from agentos.core.agent import Agent

    rag = RAGPipeline()
    rag.ingest("knowledge_base/")
    agent = Agent(tools=[create_rag_tool(rag)])
"""

from agentos.rag.chunker import Chunk, DocumentChunker
from agentos.rag.config import rag_config
from agentos.rag.embeddings import (
    BaseEmbeddings,
    EmbeddingEngine,
    LocalEmbeddings,
    OpenAIEmbeddings,
    TFIDFEmbeddings,
    get_embeddings,
)
from agentos.rag.ingestion import IngestionPipeline
from agentos.rag.pipeline import RAGPipeline, RetrievalResult
from agentos.rag.reranker import CrossEncoderReranker
from agentos.rag.retriever import HybridRetriever
from agentos.rag.types import SearchResult
from agentos.rag.vector_store import Document, VectorStore

__all__ = [
    "BaseEmbeddings",
    "Chunk",
    "CrossEncoderReranker",
    "Document",
    "DocumentChunker",
    "EmbeddingEngine",
    "HybridRetriever",
    "IngestionPipeline",
    "LocalEmbeddings",
    "OpenAIEmbeddings",
    "RAGPipeline",
    "RetrievalResult",
    "SearchResult",
    "TFIDFEmbeddings",
    "VectorStore",
    "get_embeddings",
    "rag_config",
]

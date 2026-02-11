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

from agentos.rag.pipeline import RAGPipeline, RetrievalResult
from agentos.rag.chunker import DocumentChunker, Chunk
from agentos.rag.embeddings import EmbeddingEngine
from agentos.rag.vector_store import VectorStore, SearchResult, Document

__all__ = [
    "RAGPipeline",
    "RetrievalResult",
    "DocumentChunker",
    "Chunk",
    "EmbeddingEngine",
    "VectorStore",
    "SearchResult",
    "Document",
]

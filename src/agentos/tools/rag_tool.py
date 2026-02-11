"""RAG Tool — gives any AgentOS agent the ability to search ingested documents."""

from __future__ import annotations
from agentos.core.tool import Tool, tool
from agentos.rag.pipeline import RAGPipeline


# Module-level default pipeline (lazy-initialized)
_default_pipeline: RAGPipeline | None = None


def get_default_pipeline() -> RAGPipeline:
    """Get or create the default RAG pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline


def set_default_pipeline(pipeline: RAGPipeline) -> None:
    """Set a custom pipeline as the default for the RAG tool."""
    global _default_pipeline
    _default_pipeline = pipeline


def create_rag_tool(pipeline: RAGPipeline | None = None, top_k: int = 5) -> Tool:
    """Create a RAG search tool backed by a specific pipeline.

    Usage:
        rag = RAGPipeline()
        rag.ingest("docs/")
        rag_tool = create_rag_tool(rag)
        agent = Agent(tools=[rag_tool, ...])
    """
    rag = pipeline or get_default_pipeline()

    @tool(
        name="search_documents",
        description=(
            "Search through ingested documents to find relevant information. "
            "Use this when the user asks a question that may be answered by "
            "the uploaded/ingested documents or knowledge base. "
            "Returns the most relevant text passages with source information."
        ),
    )
    def search_documents(query: str) -> str:
        result = rag.query(query, top_k=top_k)
        if not result.results:
            return "No relevant documents found for this query."
        return result.context

    return search_documents

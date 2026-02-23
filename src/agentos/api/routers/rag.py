from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agentos.rag.ingestion import IngestionPipeline
from agentos.rag.pipeline import RAGPipeline
from agentos.rag.ingestion import _store_registry

router = APIRouter(prefix="/rag", tags=["rag"])


class IngestRequest(BaseModel):
    source: str
    collection: str = "default"
    chunk_strategy: str = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 64


class SearchRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = 5


@router.post("/ingest")
async def rag_ingest(req: IngestRequest) -> dict:
    pipeline = IngestionPipeline(
        collection_name=req.collection,
        chunk_strategy=req.chunk_strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    try:
        count = pipeline.ingest_path(req.source)
        return {"chunks_added": count, "collection": req.collection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def rag_search(req: SearchRequest) -> dict:
    from agentos.rag.ingestion import _get_store

    rag = RAGPipeline()
    rag.store = _get_store(req.collection)
    try:
        result = rag.query(req.query, top_k=req.top_k)
        return {
            "query": result.query,
            "context": result.context,
            "results": [
                {"text": r.text[:500], "score": r.score, "doc_id": r.doc_id}
                for r in result.results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def rag_collections() -> list:
    return list(_store_registry.keys())

from __future__ import annotations
import os
import uuid as _uuid
from pathlib import Path as _Path
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agentos.core.multimodal import analyze_image, read_document
from agentos.monitor.store import store
from agentos.web.deps import get_upload_dir

router = APIRouter(tags=["multimodal"])

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
ALLOWED_DOC_EXTS = {".txt", ".md", ".markdown", ".pdf", ".csv", ".json", ".log", ".rst"}
ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_DOC_EXTS
class AnalyzeFileRequest(BaseModel):
    file_path: str
    question: str = ""
    model: str = "gpt-4o"
@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an image or document for analysis."""
    if not file.filename:
        return JSONResponse({"status": "error", "message": "No file provided"}, 400)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return JSONResponse(
            {
                "status": "error",
                "message": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
            },
            400,
        )

    unique_name = f"{_uuid.uuid4().hex[:8]}_{file.filename}"
    dest = get_upload_dir() / unique_name
    content = await file.read()
    dest.write_bytes(content)

    file_type = "image" if ext in ALLOWED_IMAGE_EXTS else "document"

    return {
        "status": "uploaded",
        "file_path": str(dest),
        "file_name": file.filename,
        "file_type": file_type,
        "size_bytes": len(content),
    }


class AnalyzeFileRequest(BaseModel):
    file_path: str
    question: str = ""
    model: str = "gpt-4o"


@router.post("/api/analyze-file")
def analyze_uploaded_file(req: AnalyzeFileRequest):
    """Analyze an uploaded image or document.

    For images: uses OpenAI Vision API.
    For documents: reads content and uses an agent to answer the question.
    Also accepts image URLs directly.
    """
    question = req.question.strip() or "Describe or summarize this content in detail."

    from agentos.core.multimodal import is_url

    if is_url(req.file_path):
        try:
            result = analyze_image(
                image_path_or_url=req.file_path,
                prompt=question,
                model=req.model,
            )
            return {"status": "ok", "type": "image", "analysis": result}
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, 500)

    path = _Path(req.file_path)
    if not path.exists():
        return JSONResponse({"status": "error", "message": "File not found"}, 404)

    ext = path.suffix.lower()

    if ext in ALLOWED_IMAGE_EXTS:
        try:
            result = analyze_image(
                image_path_or_url=str(path),
                prompt=question,
                model=req.model,
            )
            return {"status": "ok", "type": "image", "analysis": result}
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, 500)
    else:
        try:
            content = read_document(str(path), max_chars=30_000)
            from agentos.core.agent import Agent

            agent = Agent(
                name="doc-analyzer",
                model=req.model if req.model != "gpt-4o" else "gpt-4o-mini",
                system_prompt=(
                    "You are a document analysis assistant. The user has uploaded a document "
                    "and wants you to analyze it. Answer their question based solely on the "
                    "document content provided."
                ),
            )
            import io
            import sys

            old = sys.stdout
            sys.stdout = io.StringIO()
            msg = agent.run(
                f"Here is the document content:\n\n---\n{content}\n---\n\n"
                f"Question: {question}"
            )
            sys.stdout = old
            for e in agent.events:
                store.log_event(e)
            cost = sum(e.cost_usd for e in agent.events)
            tokens = sum(e.tokens_used for e in agent.events)
            return {
                "status": "ok",
                "type": "document",
                "analysis": msg.content,
                "cost": round(cost, 6),
                "tokens": tokens,
            }
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, 500)


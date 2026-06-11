from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from agentos.learning.feedback import (
    FeedbackEntry as _FBEntry,
    FeedbackType as _FBType,
    get_feedback_store as _get_fb,
)
from agentos.learning.analyzer import FeedbackAnalyzer as _FBAnalyzer
from agentos.learning.prompt_optimizer import PromptOptimizer as _PromptOpt
from agentos.learning.few_shot import FewShotBuilder as _FSBuilder
from agentos.learning.report import build_learning_report as _build_lr

router = APIRouter(tags=["learning"])

_prompt_optimizer: _PromptOpt | None = None
_fs_builder: _FSBuilder | None = None

class _FeedbackBody(BaseModel):
    query: str
    response: str = ""
    feedback_type: str = "thumbs_up"
    rating: float = 0
    correction: str = ""
    comment: str = ""
    topic: str = ""
    agent_name: str = ""
@router.post("/api/learning/feedback")
def learning_feedback(body: _FeedbackBody) -> dict:
    store = _get_fb()
    entry = _FBEntry(
        feedback_type=_FBType(body.feedback_type),
        query=body.query,
        response=body.response,
        rating=body.rating,
        correction=body.correction,
        comment=body.comment,
        topic=body.topic,
        agent_name=body.agent_name,
    )
    store.add(entry)
    return {"status": "ok", "id": entry.id}


@router.get("/api/learning/stats")
def learning_stats() -> dict:
    return _get_fb().stats()


@router.get("/api/learning/recent")
def learning_recent() -> list:
    return [e.model_dump() for e in _get_fb().recent(20)]


@router.get("/api/learning/analyze")
def learning_analyze() -> dict:
    analyzer = _FBAnalyzer(_get_fb())
    return analyzer.analyze().to_dict()


@router.post("/api/learning/optimize")
def learning_optimize() -> dict:
    global _prompt_optimizer
    _prompt_optimizer = _PromptOpt(_get_fb(), use_llm=False)
    patches = _prompt_optimizer.optimize()
    return {"patches": [p.to_dict() for p in patches]}


@router.post("/api/learning/few-shot")
def learning_few_shot() -> dict:
    global _fs_builder
    _fs_builder = _FSBuilder(_get_fb(), max_examples=6)
    examples = _fs_builder.build()
    return {"examples": [e.to_dict() for e in examples], "stats": _fs_builder.stats()}


@router.get("/api/learning/progress")
def learning_progress() -> dict:
    return _build_lr(_get_fb(), period="week").to_dict()


from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agentos.marketplace import get_marketplace_store

router = APIRouter(tags=["marketplace"])

class PublishRequest(BaseModel):
    name: str
    description: str = ""
    author: str = "anonymous"
    version: str = "1.0.0"
    category: str = "general"
    icon: str = ""
    tags: list[str] = []
    price: float = 0.0
    config: dict = {}


class ReviewRequest(BaseModel):
    user: str = "anonymous"
    rating: float = 5.0
    comment: str = ""
@router.get("/api/marketplace/list")
def marketplace_list(category: str = "", sort: str = "downloads"):
    """List marketplace agents, optionally filtered by category."""
    mp = get_marketplace_store()
    agents = mp.search(category=category, sort_by=sort)
    return {
        "agents": [a.to_card() for a in agents],
        "categories": mp.get_categories(),
        "stats": mp.stats(),
    }


@router.get("/api/marketplace/search")
def marketplace_search(q: str = "", category: str = "", sort: str = "downloads"):
    """Search the marketplace."""
    mp = get_marketplace_store()
    agents = mp.search(query=q, category=category, sort_by=sort)
    return {"agents": [a.to_card() for a in agents], "query": q}


@router.get("/api/marketplace/trending")
def marketplace_trending():
    mp = get_marketplace_store()
    return {"agents": [a.to_card() for a in mp.get_trending()]}


@router.get("/api/marketplace/top-rated")
def marketplace_top_rated():
    mp = get_marketplace_store()
    return {"agents": [a.to_card() for a in mp.get_top_rated()]}


@router.get("/api/marketplace/{agent_id}")
def marketplace_detail(agent_id: str):
    """Get full details for a marketplace agent, including reviews."""
    mp = get_marketplace_store()
    agent = mp.get(agent_id)
    if not agent:
        return JSONResponse({"status": "error", "message": "Agent not found"}, 404)
    data = agent.model_dump()
    data["status"] = "ok"
    return data


@router.post("/api/marketplace/publish")
def marketplace_publish(req: PublishRequest):
    """Publish a new agent to the marketplace."""
    mp = get_marketplace_store()
    agent = mp.publish(
        name=req.name,
        description=req.description,
        author=req.author,
        version=req.version,
        category=req.category,
        icon=req.icon,
        tags=req.tags,
        price=req.price,
        config=req.config,
    )
    return {"status": "published", "agent": agent.to_card()}


@router.post("/api/marketplace/install/{agent_id}")
def marketplace_install(agent_id: str):
    """Install an agent — increments download counter and returns config."""
    mp = get_marketplace_store()
    agent = mp.install(agent_id)
    if not agent:
        return JSONResponse({"status": "error", "message": "Agent not found"}, 404)
    return {
        "status": "installed",
        "agent": agent.to_card(),
        "config": agent.config.model_dump(),
    }


@router.post("/api/marketplace/review/{agent_id}")
def marketplace_review(agent_id: str, req: ReviewRequest):
    """Leave a review for a marketplace agent."""
    mp = get_marketplace_store()
    review = mp.review(agent_id, user=req.user, rating=req.rating, comment=req.comment)
    if not review:
        return JSONResponse({"status": "error", "message": "Agent not found"}, 404)
    agent = mp.get(agent_id)
    return {
        "status": "reviewed",
        "review": review.model_dump(),
        "new_rating": agent.rating if agent else 0,
        "review_count": agent.review_count if agent else 0,
    }


@router.delete("/api/marketplace/{agent_id}")
def marketplace_delete(agent_id: str):
    mp = get_marketplace_store()
    if mp.delete(agent_id):
        return {"status": "deleted"}
    return JSONResponse({"status": "error", "message": "Agent not found"}, 404)


@router.get("/marketplace/search")
def marketplace_package_search(tags: str = "", capability: str = ""):
    from agentos.marketplace.registry import MarketplaceRegistry

    reg = MarketplaceRegistry()
    return {"packages": reg.search(tags=tags, capability=capability)}


from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agentos.marketplace.registry import MarketplaceRegistry, publish, install

router = APIRouter(prefix="/marketplace", tags=["marketplace"])


class PublishRequest(BaseModel):
    manifest_path: str


class InstallRequest(BaseModel):
    name: str
    version: str | None = None


@router.get("/search")
async def marketplace_search(tags: str = "", capability: str = "") -> list:
    registry = MarketplaceRegistry()
    return registry.search(tags=tags, capability=capability)


@router.post("/publish")
async def marketplace_publish(req: PublishRequest) -> dict:
    try:
        manifest = publish(req.manifest_path)
        return manifest.model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/install")
async def marketplace_install(req: InstallRequest) -> dict:
    result = install(req.name, req.version)
    if result is None:
        raise HTTPException(status_code=404, detail="Package not found")
    return result


@router.get("/packages")
async def marketplace_packages() -> list:
    registry = MarketplaceRegistry()
    return registry.list_packages()

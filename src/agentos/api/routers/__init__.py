from __future__ import annotations
from agentos.api.routers.auth import auth_router
from agentos.api.routers.marketplace import router as marketplace_router
from agentos.api.routers.sandbox import router as sandbox_router
from agentos.api.routers.monitor import router as monitor_router
from agentos.api.routers.monitor import ws_monitor_router
from agentos.api.routers.scheduler import router as scheduler_router
from agentos.api.routers.deploy import router as deploy_router
from agentos.api.routers.teams import router as teams_router
from agentos.api.routers.rag import router as rag_router
from agentos.api.routers.compliance import router as compliance_router
from agentos.api.routers.mesh import router as mesh_router

ALL_ROUTERS = [
    (auth_router, ""),
    (marketplace_router, "/api"),
    (sandbox_router, "/api"),
    (monitor_router, "/api"),
    (ws_monitor_router, "/ws"),
    (scheduler_router, "/api"),
    (deploy_router, "/api"),
    (teams_router, "/api"),
    (rag_router, "/api"),
    (compliance_router, "/api"),
    (mesh_router, "/api"),
]

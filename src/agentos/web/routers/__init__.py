from __future__ import annotations
from agentos.web.routers.pages import router as pages_router
from agentos.web.routers.ws import router as ws_router
from agentos.web.routers.dashboard import router as dashboard_router
from agentos.web.routers.agents import router as agents_router
from agentos.web.routers.scheduler import router as scheduler_router
from agentos.web.routers.auth import router as auth_router
from agentos.web.routers.multimodal import router as multimodal_router
from agentos.web.routers.branching import router as branching_router
from agentos.web.routers.events import router as events_router
from agentos.web.routers.marketplace import router as marketplace_router
from agentos.web.routers.workflows import router as workflows_router
from agentos.web.routers.embed import router as embed_router
from agentos.web.routers.observability import router as observability_router
from agentos.web.routers.learning import router as learning_router
from agentos.web.routers.simulation import router as simulation_router
from agentos.web.routers.mesh import router as mesh_router
from agentos.web.routers.analytics import router as analytics_router

ALL_WEB_ROUTERS = [
    pages_router,
    ws_router,
    dashboard_router,
    agents_router,
    scheduler_router,
    auth_router,
    multimodal_router,
    branching_router,
    events_router,
    marketplace_router,
    workflows_router,
    embed_router,
    observability_router,
    learning_router,
    simulation_router,
    mesh_router,
    analytics_router,
]

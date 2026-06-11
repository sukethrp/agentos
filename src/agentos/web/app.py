"""AgentOS Web UI — Visual Agent Builder + Marketplace + Dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agentos.api.routers import ALL_ROUTERS
from agentos.auth.middleware import ScopeMiddleware
from agentos.demo import is_demo_mode
from agentos.web.deps import set_app
from agentos.web.routers import ALL_WEB_ROUTERS

load_dotenv()

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="AgentOS Platform", version="0.3.0", lifespan=lifespan)
set_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ScopeMiddleware)

for router, prefix in ALL_ROUTERS:
    app.include_router(router, prefix=prefix)

for router in ALL_WEB_ROUTERS:
    app.include_router(router)

if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

if is_demo_mode():
    from agentos.demo.seed import seed_all

    seed_all()

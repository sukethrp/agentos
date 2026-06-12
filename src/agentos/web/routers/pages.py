from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from agentos.demo import is_demo_mode

router = APIRouter(tags=["pages"])

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(
        _STATIC_DIR / "favicon.svg",
        media_type="image/svg+xml",
    )


@router.get("/")
def home():
    html = (_STATIC_DIR / "index.html").read_text()
    if is_demo_mode():
        banner = (_STATIC_DIR / "demo-banner.html").read_text()
        html = html.replace(
            '<div class="app">',
            banner + '<div class="app">',
        )
    return HTMLResponse(html)

from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from agentos.embed.widget import generate_widget, generate_widget_js, generate_snippet

router = APIRouter(tags=["embed"])
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class EmbedConfigRequest(BaseModel):
    agent_name: str = "AgentOS"
    theme: str = "dark"
    position: str = "bottom-right"
    accent_color: str = "#6c5ce7"
    logo: str = ""
    greeting: str = "Hi! How can I help you today?"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    tools: list[str] = []
    api_key: str = ""


@router.get("/embed/chat.js")
def embed_chat_js():
    """Serve the embeddable widget JavaScript."""
    js = generate_widget_js()
    return PlainTextResponse(js, media_type="application/javascript")


@router.get("/api/embed/widget")
def embed_widget_get(
    agent_name: str = "AgentOS",
    theme: str = "dark",
    position: str = "bottom-right",
    accent_color: str = "#6c5ce7",
    greeting: str = "Hi! How can I help you today?",
    model: str = "gpt-4o-mini",
):
    """Return a self-contained HTML widget snippet."""
    html = generate_widget(
        agent_name=agent_name,
        base_url="",  # empty = same origin
        theme=theme,
        position=position,
        accent_color=accent_color,
        greeting=greeting,
        model=model,
    )
    return HTMLResponse(html)


@router.post("/api/embed/widget")
def embed_widget_post(req: EmbedConfigRequest):
    """Return a self-contained HTML widget snippet (POST with full config)."""
    html = generate_widget(
        agent_name=req.agent_name,
        base_url="",
        api_key=req.api_key,
        theme=req.theme,
        position=req.position,
        accent_color=req.accent_color,
        greeting=req.greeting,
        model=req.model,
        system_prompt=req.system_prompt,
        tools=req.tools,
    )
    return {"status": "ok", "html": html}


@router.get("/api/embed/snippet")
def embed_snippet(
    base_url: str = "http://localhost:8000",
    agent_name: str = "AgentOS",
    theme: str = "dark",
    position: str = "bottom-right",
    accent_color: str = "#6c5ce7",
    api_key: str = "",
):
    """Return a copy-paste code snippet for embedding."""
    snippet = generate_snippet(
        base_url=base_url,
        api_key=api_key,
        agent_name=agent_name,
        theme=theme,
        position=position,
        accent_color=accent_color,
    )
    return {"status": "ok", "snippet": snippet}

@router.get("/embed/preview")
def embed_preview(
    agent_name: str = "AgentOS",
    theme: str = "dark",
    accent_color: str = "#6c5ce7",
):
    """Render a full HTML page with the widget embedded — handy for previewing."""
    widget_html = generate_widget(
        agent_name=agent_name,
        base_url="",
        theme=theme,
        accent_color=accent_color,
    )
    body_bg = "#f5f5f5" if theme == "light" else "#1a1a2e"
    body_color = "#333" if theme == "light" else "#eee"
    page = (_TEMPLATES_DIR / "embed-preview.html").read_text()
    page = page.replace("{{body_bg}}", body_bg)
    page = page.replace("{{body_color}}", body_color)
    page = page.replace("{{widget_html}}", widget_html)
    return HTMLResponse(page)


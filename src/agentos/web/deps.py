from __future__ import annotations
import os
import tempfile
import uuid as _uuid
from pathlib import Path as _Path
from fastapi import FastAPI
from agentos.events import WebhookTrigger
from agentos.scheduler import get_scheduler

_app: FastAPI | None = None
_scheduler = get_scheduler()
_webhook_trigger = WebhookTrigger(name="web-webhook")
_webhook_trigger.start()
_UPLOAD_DIR = _Path(tempfile.gettempdir()) / "agentos_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_workflows_store: dict[str, dict] = {}


def set_app(app: FastAPI) -> None:
    global _app
    _app = app


def get_app() -> FastAPI:
    if _app is None:
        raise RuntimeError("FastAPI app not initialized")
    return _app


def get_scheduler():
    return _scheduler


def get_webhook_trigger():
    return _webhook_trigger


def get_upload_dir():
    return _UPLOAD_DIR


def get_workflows_store():
    return _workflows_store

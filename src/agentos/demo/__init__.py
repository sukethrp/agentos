"""AgentOS demo mode — run the full platform without API keys."""

from __future__ import annotations

import os


def is_demo_mode() -> bool:
    """Return True when the user opted into demo mode."""
    return os.getenv("AGENTOS_DEMO_MODE", "").lower() in ("1", "true", "yes")

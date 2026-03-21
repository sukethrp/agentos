"""Launch the AgentOS Web Platform.

Demo mode (no API keys required):
    AGENTOS_DEMO_MODE=true python examples/run_web_builder.py
"""

import os
import sys
sys.path.insert(0, "src")

import uvicorn
from agentos.web.app import app

if __name__ == "__main__":
    demo = os.getenv("AGENTOS_DEMO_MODE", "").lower() in ("1", "true", "yes")

    print("=" * 60)
    print("🚀 AgentOS Web Platform")
    if demo:
        print("   ⚡ DEMO MODE — no API keys required")
    print("=" * 60)
    print()
    print("   🛠️  Agent Builder:  http://localhost:8000")
    print("   📦 Templates:      http://localhost:8000 → Templates")
    print("   💬 Chat:           http://localhost:8000 → Chat")
    print("   📊 Monitor:        http://localhost:8000 → Monitor")
    print("   🏪 Marketplace:    http://localhost:8000 → Marketplace")
    print()
    if not demo:
        print("   💡 Tip: run with AGENTOS_DEMO_MODE=true to try without API keys")
    print("   Press Ctrl+C to stop")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
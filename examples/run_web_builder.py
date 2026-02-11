"""Launch the AgentOS Web Platform."""

import sys
sys.path.insert(0, "src")

import uvicorn
from agentos.web.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AgentOS Web Platform")
    print("=" * 60)
    print()
    print("   🛠️  Agent Builder:  http://localhost:8000")
    print("   📦 Templates:      http://localhost:8000 → Templates")
    print("   💬 Chat:           http://localhost:8000 → Chat")
    print("   📊 Monitor:        http://localhost:8000 → Monitor")
    print("   🏪 Marketplace:    http://localhost:8000 → Marketplace")
    print()
    print("   Press Ctrl+C to stop")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
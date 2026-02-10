from __future__ import annotations
from agentos.core.types import AgentEvent
import time


class AgentStore:
    """In-memory store for all agent events and metrics."""

    def __init__(self):
        self.events: list[dict] = []
        self.agents: dict[str, dict] = {}

    def log_event(self, event: AgentEvent):
        """Store an agent event."""
        e = event.model_dump()
        self.events.append(e)

        name = event.agent_name
        if name not in self.agents:
            self.agents[name] = {
                "name": name,
                "status": "running",
                "total_events": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_tool_calls": 0,
                "total_llm_calls": 0,
                "last_active": time.time(),
                "quality_scores": [],
            }

        a = self.agents[name]
        a["total_events"] += 1
        a["total_cost"] += event.cost_usd
        a["total_tokens"] += event.tokens_used
        a["last_active"] = time.time()

        if event.event_type == "tool_call":
            a["total_tool_calls"] += 1
        elif event.event_type == "llm_call":
            a["total_llm_calls"] += 1

    def log_quality(self, agent_name: str, score: float):
        """Track quality score for drift detection."""
        if agent_name in self.agents:
            self.agents[agent_name]["quality_scores"].append({
                "score": score,
                "timestamp": time.time(),
            })

    def get_overview(self) -> dict:
        """Get overview of all agents."""
        total_cost = sum(a["total_cost"] for a in self.agents.values())
        total_events = sum(a["total_events"] for a in self.agents.values())
        return {
            "total_agents": len(self.agents),
            "total_events": total_events,
            "total_cost": round(total_cost, 6),
            "agents": list(self.agents.values()),
        }

    def get_agent(self, name: str) -> dict | None:
        """Get details for a single agent."""
        return self.agents.get(name)

    def get_events(self, agent_name: str | None = None, limit: int = 50) -> list[dict]:
        """Get recent events, optionally filtered by agent."""
        if agent_name:
            filtered = [e for e in self.events if e["agent_name"] == agent_name]
        else:
            filtered = self.events
        return filtered[-limit:]

    def detect_drift(self, agent_name: str, window: int = 5) -> dict | None:
        """Detect if agent quality is drifting down."""
        agent = self.agents.get(agent_name)
        if not agent or len(agent["quality_scores"]) < window * 2:
            return None

        scores = [s["score"] for s in agent["quality_scores"]]
        old_avg = sum(scores[-window*2:-window]) / window
        new_avg = sum(scores[-window:]) / window
        drift = new_avg - old_avg

        if drift < -1.0:
            return {
                "agent": agent_name,
                "alert": "QUALITY_DRIFT",
                "old_avg": round(old_avg, 1),
                "new_avg": round(new_avg, 1),
                "drift": round(drift, 1),
                "message": f"Quality dropped from {old_avg:.1f} to {new_avg:.1f}",
            }
        return None


# Global store instance
store = AgentStore()
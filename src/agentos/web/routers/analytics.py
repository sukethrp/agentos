from __future__ import annotations
from fastapi import APIRouter, Query
from agentos.monitor.store import store

router = APIRouter(tags=["analytics"])

def _bucket_key(ts: float, granularity: str) -> str:
    """Convert a unix timestamp to a bucket key string."""
    import datetime

    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    if granularity == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    elif granularity == "week":
        # ISO week start (Monday)
        start = dt - datetime.timedelta(days=dt.weekday())
        return start.strftime("%Y-%m-%d")
    else:  # day (default)
        return dt.strftime("%Y-%m-%d")
@router.get("/api/analytics/cost-over-time")
def analytics_cost_over_time(
    granularity: str = Query("day", pattern="^(hour|day|week)$"),
):
    """Aggregate cost by time bucket (hour / day / week)."""
    buckets: dict[str, dict] = {}
    for ev in store.events:
        key = _bucket_key(ev["timestamp"], granularity)
        if key not in buckets:
            buckets[key] = {"bucket": key, "cost": 0.0, "tokens": 0, "queries": 0}
        buckets[key]["cost"] += ev.get("cost_usd", 0.0)
        buckets[key]["tokens"] += ev.get("tokens_used", 0)
        if ev.get("event_type") == "llm_call":
            buckets[key]["queries"] += 1
    series = sorted(buckets.values(), key=lambda b: b["bucket"])
    for b in series:
        b["cost"] = round(b["cost"], 6)
    return {"granularity": granularity, "series": series}


@router.get("/api/analytics/popular-tools")
def analytics_popular_tools():
    """Rank tools by usage count."""
    counts: dict[str, dict] = {}
    for ev in store.events:
        if ev.get("event_type") != "tool_call":
            continue
        tool_name = (ev.get("data") or {}).get("tool", "unknown")
        if tool_name not in counts:
            counts[tool_name] = {
                "tool": tool_name,
                "count": 0,
                "total_latency_ms": 0.0,
                "total_cost": 0.0,
            }
        counts[tool_name]["count"] += 1
        counts[tool_name]["total_latency_ms"] += ev.get("latency_ms", 0.0)
        counts[tool_name]["total_cost"] += ev.get("cost_usd", 0.0)
    ranked = sorted(counts.values(), key=lambda t: t["count"], reverse=True)
    for t in ranked:
        t["avg_latency_ms"] = round(t["total_latency_ms"] / max(t["count"], 1), 1)
        t["total_cost"] = round(t["total_cost"], 6)
    return {"tools": ranked}


@router.get("/api/analytics/model-comparison")
def analytics_model_comparison():
    """Compare models by cost, speed, tokens, and call count."""
    models: dict[str, dict] = {}
    for ev in store.events:
        if ev.get("event_type") != "llm_call":
            continue
        model = (ev.get("data") or {}).get("model", "unknown")
        if model not in models:
            models[model] = {
                "model": model,
                "calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_latency_ms": 0.0,
                "quality_scores": [],
            }
        m = models[model]
        m["calls"] += 1
        m["total_cost"] += ev.get("cost_usd", 0.0)
        m["total_tokens"] += ev.get("tokens_used", 0)
        m["total_latency_ms"] += ev.get("latency_ms", 0.0)
    for agent_data in store.agents.values():
        for qs in agent_data.get("quality_scores", []):
            # Attempt to attribute to last-used model (best effort)
            pass
    result = []
    for m in models.values():
        m["avg_cost"] = round(m["total_cost"] / max(m["calls"], 1), 6)
        m["avg_latency_ms"] = round(m["total_latency_ms"] / max(m["calls"], 1), 1)
        m["avg_tokens"] = round(m["total_tokens"] / max(m["calls"], 1))
        m["total_cost"] = round(m["total_cost"], 6)
        result.append(m)
    result.sort(key=lambda x: x["calls"], reverse=True)
    return {"models": result}


@router.get("/api/analytics/agent-leaderboard")
def analytics_agent_leaderboard():
    """Rank agents by quality, cost-efficiency, and usage."""
    leaderboard = []
    for name, a in store.agents.items():
        scores = [s["score"] for s in a.get("quality_scores", [])]
        avg_quality = round(sum(scores) / len(scores), 2) if scores else None
        total_queries = a.get("total_llm_calls", 0)
        cost_per_query = round(a["total_cost"] / max(total_queries, 1), 6)
        leaderboard.append(
            {
                "agent": name,
                "avg_quality": avg_quality,
                "total_cost": round(a["total_cost"], 6),
                "total_tokens": a["total_tokens"],
                "total_queries": total_queries,
                "total_tool_calls": a.get("total_tool_calls", 0),
                "cost_per_query": cost_per_query,
                "total_events": a["total_events"],
            }
        )
    leaderboard.sort(
        key=lambda x: (x["avg_quality"] or 0, x["total_queries"]), reverse=True
    )
    total_spend = round(sum(a["total_cost"] for a in store.agents.values()), 6)
    total_queries = sum(a.get("total_llm_calls", 0) for a in store.agents.values())
    avg_cost = round(total_spend / max(total_queries, 1), 6)
    return {
        "leaderboard": leaderboard,
        "summary": {
            "total_spend": total_spend,
            "total_queries": total_queries,
            "avg_cost_per_query": avg_cost,
            "total_agents": len(store.agents),
            "total_events": len(store.events),
        },
    }


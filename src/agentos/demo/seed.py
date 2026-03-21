"""Seed the monitor store and marketplace with realistic sample data.

Called once on startup when ``AGENTOS_DEMO_MODE=true``.
"""

from __future__ import annotations

import random
import time

from agentos.core.types import AgentEvent


# ---------------------------------------------------------------------------
# Monitor seed data
# ---------------------------------------------------------------------------

_SAMPLE_AGENTS = [
    {
        "name": "customer-support-bot",
        "events": [
            ("llm_call", 320, 1_200, 0.0024),
            ("tool_call", 45, 0, 0.0),
            ("llm_call", 280, 950, 0.0019),
            ("llm_call", 410, 1_600, 0.0032),
            ("tool_call", 52, 0, 0.0),
            ("llm_call", 350, 1_100, 0.0022),
            ("tool_call", 38, 0, 0.0),
            ("llm_call", 290, 900, 0.0018),
        ],
    },
    {
        "name": "research-assistant",
        "events": [
            ("llm_call", 520, 2_100, 0.0042),
            ("tool_call", 1_200, 0, 0.0),
            ("tool_call", 890, 0, 0.0),
            ("llm_call", 680, 2_800, 0.0056),
            ("llm_call", 450, 1_900, 0.0038),
            ("tool_call", 650, 0, 0.0),
            ("llm_call", 390, 1_500, 0.003),
        ],
    },
    {
        "name": "code-reviewer",
        "events": [
            ("llm_call", 750, 3_200, 0.0064),
            ("llm_call", 620, 2_600, 0.0052),
            ("tool_call", 110, 0, 0.0),
            ("llm_call", 580, 2_400, 0.0048),
            ("llm_call", 490, 2_000, 0.004),
        ],
    },
    {
        "name": "sales-outreach-agent",
        "events": [
            ("llm_call", 260, 800, 0.0016),
            ("tool_call", 95, 0, 0.0),
            ("llm_call", 310, 1_000, 0.002),
            ("tool_call", 78, 0, 0.0),
            ("llm_call", 240, 750, 0.0015),
            ("llm_call", 270, 850, 0.0017),
        ],
    },
]


def seed_monitor_store() -> None:
    """Populate the global monitor store with sample agent events."""
    from agentos.monitor.store import store

    if store.events:
        return

    base_time = time.time() - 3600

    for agent_info in _SAMPLE_AGENTS:
        name = agent_info["name"]
        for i, (etype, latency, tokens, cost) in enumerate(agent_info["events"]):
            event = AgentEvent(
                agent_name=name,
                event_type=etype,
                timestamp=base_time + i * random.uniform(30, 120),
                tokens_used=tokens,
                cost_usd=cost,
                latency_ms=latency + random.uniform(-20, 20),
                data={
                    "provider": "demo",
                    "model": "gpt-4o-mini" if "llm" in etype else "",
                },
            )
            store.log_event(event)

        for _ in range(random.randint(3, 8)):
            store.log_quality(name, round(random.uniform(6.5, 9.5), 1))


# ---------------------------------------------------------------------------
# Marketplace seed data
# ---------------------------------------------------------------------------

_SAMPLE_MARKETPLACE_AGENTS = [
    {
        "name": "Customer Support Pro",
        "description": "Production-ready customer support agent with FAQ lookup, "
        "ticket creation, and sentiment analysis. Handles returns, billing, "
        "and technical issues.",
        "author": "AgentOS Team",
        "category": "support",
        "icon": "🎧",
        "tags": ["customer-support", "faq", "tickets", "production-ready"],
        "downloads": 1_842,
        "rating": 4.6,
        "reviews": [
            ("alice", 5.0, "Saved us weeks of work. Tool routing is excellent."),
            ("bob_dev", 4.0, "Good baseline, needed some prompt tuning for our domain."),
            ("carol", 4.8, "Great out of the box. The sentiment analysis tool is handy."),
        ],
        "config": {
            "name": "support-agent",
            "model": "gpt-4o-mini",
            "system_prompt": "You are a helpful customer support agent. Be empathetic, accurate, and concise.",
            "tools": ["web_search", "calculator"],
            "temperature": 0.3,
        },
    },
    {
        "name": "Research Assistant",
        "description": "Multi-source research agent that searches the web, "
        "summarizes documents, and synthesizes findings into structured reports.",
        "author": "AgentOS Team",
        "category": "research",
        "icon": "🔬",
        "tags": ["research", "web-search", "summarization", "reports"],
        "downloads": 3_217,
        "rating": 4.8,
        "reviews": [
            ("researcher1", 5.0, "Best research agent template I've found."),
            ("data_sarah", 4.5, "Excellent for literature reviews."),
        ],
        "config": {
            "name": "research-agent",
            "model": "gpt-4o",
            "system_prompt": "You are a thorough research assistant. Search multiple sources, cross-reference, cite findings.",
            "tools": ["web_search", "read_document", "calculator"],
            "temperature": 0.5,
        },
    },
    {
        "name": "Code Reviewer",
        "description": "Automated code review agent that checks for bugs, "
        "security issues, style violations, and suggests improvements.",
        "author": "dev_tools_inc",
        "category": "developer-tools",
        "icon": "🔍",
        "tags": ["code-review", "security", "best-practices", "developer"],
        "downloads": 2_456,
        "rating": 4.3,
        "reviews": [
            ("senior_dev", 4.5, "Catches things our linter misses. Good security checks."),
            ("junior_coder", 4.0, "Very helpful for learning best practices."),
        ],
        "config": {
            "name": "code-reviewer",
            "model": "gpt-4o",
            "system_prompt": "You are an expert code reviewer. Focus on bugs, security, readability, and performance.",
            "tools": ["web_search"],
            "temperature": 0.2,
        },
    },
    {
        "name": "Sales Outreach Agent",
        "description": "Generates personalized outreach emails, follow-ups, "
        "and meeting summaries. Includes company research and CRM integration.",
        "author": "growth_team",
        "category": "sales",
        "icon": "📧",
        "tags": ["sales", "email", "outreach", "crm"],
        "downloads": 967,
        "rating": 4.1,
        "reviews": [
            ("sales_lead", 4.0, "Good personalization. Needs more CRM connectors."),
        ],
        "config": {
            "name": "sales-agent",
            "model": "gpt-4o-mini",
            "system_prompt": "You are a sales outreach specialist. Research prospects, personalize messages, be professional.",
            "tools": ["web_search", "company_lookup"],
            "temperature": 0.6,
        },
    },
    {
        "name": "Data Analyst",
        "description": "Analyzes CSV data, generates insights, creates "
        "statistical summaries, and answers natural-language data questions.",
        "author": "analytics_lab",
        "category": "data",
        "icon": "📊",
        "tags": ["data-analysis", "csv", "statistics", "insights"],
        "downloads": 1_523,
        "rating": 4.5,
        "reviews": [
            ("data_eng", 4.5, "Handles messy CSVs surprisingly well."),
            ("pm_mike", 4.5, "Non-technical team members love this."),
        ],
        "config": {
            "name": "data-analyst",
            "model": "gpt-4o",
            "system_prompt": "You are a data analyst. Parse data, compute statistics, explain findings clearly.",
            "tools": ["calculator", "read_document"],
            "temperature": 0.3,
        },
    },
]


def seed_marketplace() -> None:
    """Populate the marketplace with sample agent templates."""
    from agentos.marketplace.store import get_marketplace_store
    from agentos.marketplace.models import AgentConfig

    mkt = get_marketplace_store()

    if mkt.list_all():
        return

    for info in _SAMPLE_MARKETPLACE_AGENTS:
        agent = mkt.publish(
            name=info["name"],
            description=info["description"],
            author=info["author"],
            category=info["category"],
            icon=info["icon"],
            tags=info["tags"],
            config=AgentConfig.model_validate(info["config"]),
        )

        agent.downloads = info["downloads"]
        agent.rating = info["rating"]

        for user, rating, comment in info.get("reviews", []):
            agent.add_review(user=user, rating=rating, comment=comment)

        agent.rating = info["rating"]

    mkt._save()


# ---------------------------------------------------------------------------
# Combined seeder
# ---------------------------------------------------------------------------


def seed_all() -> None:
    """Seed all demo data (monitor + marketplace)."""
    seed_monitor_store()
    seed_marketplace()

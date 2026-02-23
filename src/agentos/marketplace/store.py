"""Marketplace Store — JSON-file-backed registry for published agents."""

from __future__ import annotations

import json
import os

from agentos.marketplace.models import AgentConfig, MarketplaceAgent, Review


class MarketplaceStore:
    """Persistent store for marketplace listings.

    Data lives in ``<data_dir>/marketplace.json``.
    """

    def __init__(self, data_dir: str = "./agent_data/marketplace"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, "marketplace.json")
        self._agents: dict[str, MarketplaceAgent] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    raw = json.load(f)
                for item in raw:
                    agent = MarketplaceAgent.model_validate(item)
                    self._agents[agent.id] = agent
            except (json.JSONDecodeError, Exception):
                self._agents = {}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(
                [a.model_dump() for a in self._agents.values()],
                f,
                indent=2,
                default=str,
            )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(
        self,
        name: str,
        description: str,
        author: str = "anonymous",
        version: str = "1.0.0",
        category: str = "general",
        icon: str = "🤖",
        tags: list[str] | None = None,
        price: float = 0.0,
        config: AgentConfig | dict | None = None,
    ) -> MarketplaceAgent:
        """Create a new marketplace listing and persist it."""
        if isinstance(config, dict):
            config = AgentConfig.model_validate(config)
        elif config is None:
            config = AgentConfig()

        agent = MarketplaceAgent(
            name=name,
            description=description,
            author=author,
            version=version,
            category=category,
            icon=icon,
            tags=tags or [],
            price=price,
            config=config,
        )
        self._agents[agent.id] = agent
        self._save()
        return agent

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, agent_id: str) -> MarketplaceAgent | None:
        return self._agents.get(agent_id)

    def list_all(self) -> list[MarketplaceAgent]:
        return list(self._agents.values())

    def search(
        self,
        query: str = "",
        category: str = "",
        sort_by: str = "downloads",
    ) -> list[MarketplaceAgent]:
        """Search listings by name / description / tags and optional category."""
        results = list(self._agents.values())

        if category:
            cat_lower = category.lower()
            results = [a for a in results if a.category.lower() == cat_lower]

        if query:
            q = query.lower()
            scored: list[tuple[float, MarketplaceAgent]] = []
            for a in results:
                score = 0.0
                if q in a.name.lower():
                    score += 3.0
                if q in a.description.lower():
                    score += 1.0
                for tag in a.tags:
                    if q in tag.lower():
                        score += 2.0
                if q in a.author.lower():
                    score += 0.5
                if score > 0:
                    scored.append((score, a))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [a for _, a in scored]

        if sort_by == "rating":
            results.sort(key=lambda a: a.rating, reverse=True)
        elif sort_by == "newest":
            results.sort(key=lambda a: a.published_at, reverse=True)
        else:  # downloads
            results.sort(key=lambda a: a.downloads, reverse=True)

        return results

    def get_categories(self) -> list[str]:
        """Return distinct categories in the store."""
        cats = sorted({a.category for a in self._agents.values()})
        return cats

    def get_trending(self, limit: int = 10) -> list[MarketplaceAgent]:
        """Most downloaded agents (proxy for "trending this week")."""
        return sorted(self._agents.values(), key=lambda a: a.downloads, reverse=True)[
            :limit
        ]

    def get_top_rated(self, limit: int = 10) -> list[MarketplaceAgent]:
        """Highest-rated agents with at least 1 review."""
        rated = [a for a in self._agents.values() if a.review_count > 0]
        rated.sort(key=lambda a: (a.rating, a.review_count), reverse=True)
        return rated[:limit]

    # ------------------------------------------------------------------
    # Install (bump download counter)
    # ------------------------------------------------------------------

    def install(self, agent_id: str) -> MarketplaceAgent | None:
        """Mark an agent as installed — increments downloads and returns config."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None
        agent.increment_downloads()
        self._save()
        return agent

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    def review(
        self,
        agent_id: str,
        user: str,
        rating: float,
        comment: str = "",
    ) -> Review | None:
        """Add a review to an agent listing."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None
        r = agent.add_review(user=user, rating=rating, comment=comment)
        self._save()
        return r

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        agents = list(self._agents.values())
        return {
            "total_agents": len(agents),
            "total_downloads": sum(a.downloads for a in agents),
            "total_reviews": sum(a.review_count for a in agents),
            "categories": self.get_categories(),
            "free_count": sum(1 for a in agents if a.price == 0),
            "paid_count": sum(1 for a in agents if a.price > 0),
        }


# ---------------------------------------------------------------------------
# Default singleton (lazy init)
# ---------------------------------------------------------------------------

_default_store: MarketplaceStore | None = None


def get_marketplace_store() -> MarketplaceStore:
    global _default_store
    if _default_store is None:
        _default_store = MarketplaceStore()
    return _default_store

"""Marketplace data models."""

from __future__ import annotations

import time
import uuid
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Serialisable agent configuration that can be published / installed."""

    name: str = "agent"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    tools: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_iterations: int = 10


class Review(BaseModel):
    """A user review for a marketplace listing."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    user: str = "anonymous"
    rating: float = 5.0  # 1-5
    comment: str = ""
    timestamp: float = Field(default_factory=time.time)


class MarketplaceAgent(BaseModel):
    """A published agent template in the marketplace."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str
    description: str = ""
    author: str = "anonymous"
    version: str = "1.0.0"
    category: str = "general"
    icon: str = "🤖"
    tags: list[str] = Field(default_factory=list)
    price: float = 0.0  # 0 = free
    downloads: int = 0
    rating: float = 0.0
    review_count: int = 0
    reviews: list[Review] = Field(default_factory=list)
    config: AgentConfig = Field(default_factory=AgentConfig)
    published_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def add_review(self, user: str, rating: float, comment: str) -> Review:
        """Add a review and recalculate the aggregate rating."""
        r = Review(user=user, rating=max(1.0, min(5.0, rating)), comment=comment)
        self.reviews.append(r)
        self.review_count = len(self.reviews)
        self.rating = round(
            sum(rv.rating for rv in self.reviews) / self.review_count, 2
        )
        self.updated_at = time.time()
        return r

    def increment_downloads(self) -> None:
        self.downloads += 1
        self.updated_at = time.time()

    def to_card(self) -> dict:
        """Compact representation for listing cards."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "category": self.category,
            "icon": self.icon,
            "tags": self.tags,
            "price": self.price,
            "downloads": self.downloads,
            "rating": self.rating,
            "review_count": self.review_count,
        }

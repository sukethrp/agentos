from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class PackageManifest(BaseModel):
    name: str
    version: str
    description: str = ""
    author: str = ""
    entry_point: str
    inputs: dict = {}
    outputs: dict = {}
    pricing_model: Literal["free", "usage", "subscription"] = "free"
    tags: list[str] = []
    capability: str = ""

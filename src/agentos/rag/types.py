from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    doc_id: str = ""
    index: int = 0

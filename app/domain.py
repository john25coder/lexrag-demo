from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    position: int
    content: str
    page: int | None
    embedding: list[float]
    embedding_backend: str


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    filename: str
    position: int
    content: str
    page: int | None
    score: float


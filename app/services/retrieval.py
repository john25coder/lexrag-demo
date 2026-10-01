from __future__ import annotations

from app.domain import SearchHit
from app.repository import SQLiteRepository
from app.services.embeddings import EmbeddingProvider


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class RetrievalService:
    def __init__(self, repository: SQLiteRepository, embedder: EmbeddingProvider) -> None:
        self.repository = repository
        self.embedder = embedder

    def search(self, question: str, top_k: int) -> list[SearchHit]:
        query_vector = self.embedder.embed([question])[0]
        rows = self.repository.list_chunks(self.embedder.name)
        hits = [
            SearchHit(
                chunk_id=row["id"],
                document_id=row["document_id"],
                filename=row["filename"],
                position=row["position"],
                content=row["content"],
                page=row["page"],
                score=cosine_similarity(query_vector, row["embedding"]),
            )
            for row in rows
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


from __future__ import annotations

from app.services.answering import AnswerGenerator
from app.services.retrieval import RetrievalService


class RagService:
    def __init__(
        self,
        retriever: RetrievalService,
        answer_generator: AnswerGenerator,
        min_relevance_score: float,
    ) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.min_relevance_score = min_relevance_score

    def query(self, question: str, top_k: int) -> dict:
        hits = [
            hit
            for hit in self.retriever.search(question, top_k)
            if hit.score >= self.min_relevance_score
        ]
        if not hits:
            return {
                "answer": "Nao encontrei informacao suficiente nos documentos indexados.",
                "sources": [],
            }

        answer = self.answer_generator.answer(question, hits)
        sources = [
            {
                "reference": f"F{number}",
                "document_id": hit.document_id,
                "filename": hit.filename,
                "page": hit.page,
                "chunk_position": hit.position,
                "score": round(hit.score, 4),
                "excerpt": hit.content[:600],
            }
            for number, hit in enumerate(hits, start=1)
        ]
        return {"answer": answer, "sources": sources}


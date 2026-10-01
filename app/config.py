from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path(os.getenv("RAG_DATABASE_PATH", "data/lexrag.sqlite3"))
    embedding_backend: str = os.getenv("RAG_EMBEDDING_BACKEND", "hashing")
    answer_backend: str = os.getenv("RAG_ANSWER_BACKEND", "extractive")
    sentence_model: str = os.getenv(
        "RAG_SENTENCE_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    chunk_size_words: int = int(os.getenv("RAG_CHUNK_SIZE_WORDS", "180"))
    chunk_overlap_words: int = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "35"))
    min_relevance_score: float = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.10"))


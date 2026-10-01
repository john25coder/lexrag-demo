from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

from app.config import Settings

TOKEN_PATTERN = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)


class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbeddingProvider:
    """Small, deterministic local embedding suitable for a zero-config demo.

    This model is lexical rather than semantic. It keeps the full RAG pipeline
    executable offline and makes the vectorization logic easy to explain.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing-v1-{self.dimensions}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        counts = Counter(features)
        vector = [0.0] * self.dimensions

        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = -1.0 if value & 1 else 1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(component * component for component in vector))
        return [component / norm for component in vector] if norm else vector


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Instale requirements-semantic.txt para usar embeddings semanticos."
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_backend == "hashing":
        return HashingEmbeddingProvider()
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(settings.sentence_model)
    raise ValueError(f"Backend de embeddings desconhecido: {settings.embedding_backend}")


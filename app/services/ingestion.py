from __future__ import annotations

import hashlib
import io
import re
import uuid
from pathlib import Path

from pypdf import PdfReader

from app.domain import Chunk, TextSegment
from app.repository import SQLiteRepository
from app.services.embeddings import EmbeddingProvider

SUPPORTED_TEXT_TYPES = {".txt", ".md"}
SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class UnsupportedDocumentError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class DocumentLoader:
    def load(self, filename: str, content: bytes) -> list[TextSegment]:
        suffix = Path(filename).suffix.lower()
        if suffix in SUPPORTED_TEXT_TYPES:
            return [TextSegment(self._decode_text(content))]
        if suffix == ".pdf":
            return self._load_pdf(content)
        if suffix in SUPPORTED_IMAGE_TYPES:
            return [TextSegment(self._ocr_image(content))]
        raise UnsupportedDocumentError(
            "Formato nao suportado. Use PDF, TXT, MD, PNG, JPG, JPEG, TIF ou TIFF."
        )

    @staticmethod
    def _decode_text(content: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise UnsupportedDocumentError("Nao foi possivel identificar a codificacao do arquivo.")

    @staticmethod
    def _load_pdf(content: bytes) -> list[TextSegment]:
        reader = PdfReader(io.BytesIO(content))
        segments = [
            TextSegment(text=(page.extract_text() or "").strip(), page=index)
            for index, page in enumerate(reader.pages, start=1)
        ]
        return [segment for segment in segments if segment.text]

    @staticmethod
    def _ocr_image(content: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise UnsupportedDocumentError(
                "OCR e opcional. Instale requirements-ocr.txt e o executavel Tesseract."
            ) from exc
        return pytesseract.image_to_string(Image.open(io.BytesIO(content)), lang="por")


class TextChunker:
    def __init__(self, chunk_size_words: int, overlap_words: int) -> None:
        if overlap_words >= chunk_size_words:
            raise ValueError("A sobreposicao deve ser menor que o tamanho do trecho.")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def split(self, segments: list[TextSegment]) -> list[tuple[str, int | None]]:
        chunks: list[tuple[str, int | None]] = []
        step = self.chunk_size_words - self.overlap_words
        for segment in segments:
            cleaned = re.sub(r"\s+", " ", segment.text).strip()
            words = cleaned.split()
            for start in range(0, len(words), step):
                text = " ".join(words[start : start + self.chunk_size_words]).strip()
                if text:
                    chunks.append((text, segment.page))
                if start + self.chunk_size_words >= len(words):
                    break
        return chunks


class IngestionService:
    def __init__(
        self,
        repository: SQLiteRepository,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedder: EmbeddingProvider,
    ) -> None:
        self.repository = repository
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder

    def index(self, filename: str, content: bytes) -> dict:
        sha256 = hashlib.sha256(content).hexdigest()
        existing = self.repository.find_document_by_hash(sha256)
        if existing:
            return {**existing, "duplicate": True, "chunk_count": None}

        segments = self.loader.load(filename, content)
        raw_chunks = self.chunker.split(segments)
        if not raw_chunks:
            raise EmptyDocumentError(
                "O documento nao possui texto extraivel. Para PDF digitalizado, envie imagens "
                "das paginas com o suporte opcional de OCR."
            )

        document_id = str(uuid.uuid4())
        vectors = self.embedder.embed([text for text, _ in raw_chunks])
        chunks = [
            Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                position=position,
                content=text,
                page=page,
                embedding=vector,
                embedding_backend=self.embedder.name,
            )
            for position, ((text, page), vector) in enumerate(zip(raw_chunks, vectors))
        ]
        self.repository.save_document(document_id, filename, sha256, chunks)
        return {
            "id": document_id,
            "filename": filename,
            "sha256": sha256,
            "duplicate": False,
            "chunk_count": len(chunks),
        }


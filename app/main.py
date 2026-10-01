from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.api.schemas import QueryRequest, QueryResponse
from app.config import Settings
from app.repository import SQLiteRepository
from app.services.answering import create_answer_generator
from app.services.embeddings import create_embedding_provider
from app.services.ingestion import (
    DocumentLoader,
    EmptyDocumentError,
    IngestionService,
    TextChunker,
    UnsupportedDocumentError,
)
from app.services.rag import RagService
from app.services.retrieval import RetrievalService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    repository = SQLiteRepository(settings.database_path)
    embedder = create_embedding_provider(settings)
    ingestion = IngestionService(
        repository,
        DocumentLoader(),
        TextChunker(settings.chunk_size_words, settings.chunk_overlap_words),
        embedder,
    )
    rag = RagService(
        RetrievalService(repository, embedder),
        create_answer_generator(settings),
        settings.min_relevance_score,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        repository.initialize()
        yield

    app = FastAPI(
        title="LexRAG Demo API",
        version="1.0.0",
        description=(
            "API demonstrativa para indexacao e consulta de documentos com RAG e fontes. "
            "Nao oferece aconselhamento juridico."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", tags=["sistema"])
    def health() -> dict:
        return {
            "status": "ok",
            "embedding_backend": embedder.name,
            "answer_backend": settings.answer_backend,
        }

    @app.post("/documents/index", tags=["documentos"], status_code=status.HTTP_201_CREATED)
    async def index_documents(files: list[UploadFile] = File(...)) -> dict:
        results = []
        for uploaded in files:
            try:
                results.append(ingestion.index(uploaded.filename or "documento", await uploaded.read()))
            except (UnsupportedDocumentError, EmptyDocumentError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"documents": results}

    @app.get("/documents", tags=["documentos"])
    def list_documents() -> dict:
        return {"documents": repository.list_documents()}

    @app.delete("/documents/{document_id}", tags=["documentos"], status_code=204)
    def delete_document(document_id: str) -> None:
        if not repository.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Documento nao encontrado.")

    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    def query(request: QueryRequest) -> dict:
        return rag.query(request.question, request.top_k)

    return app


app = create_app()


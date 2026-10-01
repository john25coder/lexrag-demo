from __future__ import annotations

import argparse
from pathlib import Path

from app.config import Settings
from app.repository import SQLiteRepository
from app.services.answering import create_answer_generator
from app.services.embeddings import create_embedding_provider
from app.services.ingestion import DocumentLoader, IngestionService, TextChunker
from app.services.rag import RagService
from app.services.retrieval import RetrievalService


def build_services() -> tuple[SQLiteRepository, IngestionService, RagService]:
    settings = Settings()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
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
    return repository, ingestion, rag


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(file for file in path.rglob("*") if file.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente local do LexRAG Demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Indexa um arquivo ou diretorio")
    index_parser.add_argument("path", type=Path)
    ask_parser = subparsers.add_parser("ask", help="Consulta os documentos indexados")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=4)
    subparsers.add_parser("list", help="Lista documentos indexados")
    args = parser.parse_args()

    repository, ingestion, rag = build_services()
    if args.command == "index":
        for file in collect_files(args.path):
            result = ingestion.index(file.name, file.read_bytes())
            print(f"{result['filename']}: {result.get('chunk_count')} trechos")
    elif args.command == "ask":
        result = rag.query(args.question, args.top_k)
        print(result["answer"])
        for source in result["sources"]:
            print(
                f"[{source['reference']}] {source['filename']} "
                f"(score={source['score']})"
            )
    else:
        for document in repository.list_documents():
            print(f"{document['id']} | {document['filename']} | {document['chunk_count']} trechos")


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from app.domain import Chunk


class SQLiteRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    embedding_json TEXT NOT NULL,
                    embedding_backend TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_backend ON chunks(embedding_backend);
                """
            )

    def find_document_by_hash(self, sha256: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, filename, sha256, created_at FROM documents WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        return dict(row) if row else None

    def save_document(
        self,
        document_id: str,
        filename: str,
        sha256: str,
        chunks: list[Chunk],
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO documents(id, filename, sha256, created_at) VALUES (?, ?, ?, ?)",
                (document_id, filename, sha256, datetime.now(UTC).isoformat()),
            )
            connection.executemany(
                """
                INSERT INTO chunks(
                    id, document_id, position, content, page,
                    embedding_json, embedding_backend
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.position,
                        chunk.content,
                        chunk.page,
                        json.dumps(chunk.embedding),
                        chunk.embedding_backend,
                    )
                    for chunk in chunks
                ],
            )

    def list_documents(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT d.id, d.filename, d.sha256, d.created_at, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    def list_chunks(self, embedding_backend: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.document_id, d.filename, c.position, c.content, c.page,
                       c.embedding_json
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding_backend = ?
                """,
                (embedding_backend,),
            ).fetchall()
        return [
            {**dict(row), "embedding": json.loads(row["embedding_json"])}
            for row in rows
        ]


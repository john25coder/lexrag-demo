from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_reports_active_backends(tmp_path: Path) -> None:
    app = create_app(Settings(database_path=tmp_path / "test.sqlite3"))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["embedding_backend"].startswith("hashing-v1")
    assert response.json()["answer_backend"] == "extractive"


def test_index_query_and_sources(tmp_path: Path) -> None:
    app = create_app(Settings(database_path=tmp_path / "test.sqlite3"))
    content = (
        "A Politica Ficticia Alfa determina que contratos encerrados sejam "
        "mantidos por cinco anos. O descarte deve ser registrado pelo responsavel."
    )
    with TestClient(app) as client:
        indexed = client.post(
            "/documents/index",
            files=[("files", ("politica.txt", content.encode("utf-8"), "text/plain"))],
        )
        queried = client.post(
            "/query",
            json={"question": "Por quanto tempo os contratos encerrados devem ser mantidos?"},
        )

    assert indexed.status_code == 201
    assert indexed.json()["documents"][0]["chunk_count"] == 1
    assert queried.status_code == 200
    body = queried.json()
    assert body["sources"][0]["filename"] == "politica.txt"
    assert "[F1]" in body["answer"]
    assert "cinco anos" in body["answer"]


def test_duplicate_content_is_not_reindexed(tmp_path: Path) -> None:
    app = create_app(Settings(database_path=tmp_path / "test.sqlite3"))
    file = ("files", ("nota.md", b"Conteudo ficticio para teste de duplicidade.", "text/markdown"))
    with TestClient(app) as client:
        first = client.post("/documents/index", files=[file])
        second = client.post("/documents/index", files=[file])
        documents = client.get("/documents")

    assert first.json()["documents"][0]["duplicate"] is False
    assert second.json()["documents"][0]["duplicate"] is True
    assert len(documents.json()["documents"]) == 1


def test_query_without_documents_is_explicit(tmp_path: Path) -> None:
    app = create_app(Settings(database_path=tmp_path / "test.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "Qual e o prazo?"})

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "Nao encontrei" in response.json()["answer"]


"""DELETE /documents/{id} — runs offline with a stubbed DB session."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app


class FakeSession:
    def __init__(self, document=None):
        self.document = document
        self.deleted = None
        self.committed = False

    def get(self, model, pk):
        if self.document is not None and self.document.id == pk:
            return self.document
        return None

    def delete(self, obj):
        self.deleted = obj

    def commit(self):
        self.committed = True


def _client(session):
    def _fake_db():
        yield session

    app.dependency_overrides[get_db] = _fake_db
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_delete_existing_document_returns_204():
    document = SimpleNamespace(id=7)
    session = FakeSession(document)
    response = _client(session).delete("/documents/7")
    assert response.status_code == 204
    assert session.deleted is document
    assert session.committed is True


def test_delete_missing_document_returns_404():
    session = FakeSession(None)
    response = _client(session).delete("/documents/99")
    assert response.status_code == 404
    assert session.deleted is None
    assert session.committed is False

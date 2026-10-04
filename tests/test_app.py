from fastapi.testclient import TestClient

from science_rag import events
from science_rag.workflows import FUNCTIONS


def test_all_workflows_are_registered():
    assert len(FUNCTIONS) == 4


def test_event_names_are_unique():
    names = [events.INGEST_PDF, events.INGEST_IMAGE, events.ASK_QUESTION]
    assert len(set(names)) == len(names)


def test_health_reports_ok_when_database_is_up(monkeypatch):
    from science_rag import app as app_module

    class Store:
        @staticmethod
        def ping():
            return True

    class Service:
        text_store = Store()

    monkeypatch.setattr(app_module, "get_service", lambda: Service())
    response = TestClient(app_module.app).get("/health")
    assert response.json() == {"status": "ok", "qdrant": "up"}


def test_health_degrades_instead_of_crashing(monkeypatch):
    from science_rag import app as app_module

    def broken():
        raise RuntimeError("no database")

    monkeypatch.setattr(app_module, "get_service", broken)
    response = TestClient(app_module.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"

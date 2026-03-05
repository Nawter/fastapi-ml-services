import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_TICKET = (
    "I ordered a laptop from your London store last Tuesday. "
    "The device was delivered by John Smith but arrived with a cracked screen. "
    "I contacted Apple support but they said the damage was caused during shipping. "
    "This is completely unacceptable and I want a full refund immediately."
)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["all_loaded"] is True
    assert "summariser" in body["models"]
    assert "sentiment" in body["models"]
    assert "ner" in body["models"]


def test_process_returns_all_fields():
    response = client.post("/process", json={"text": SAMPLE_TICKET})
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "sentiment" in body
    assert "sentiment_score" in body
    assert "entities" in body
    assert isinstance(body["entities"], list)


def test_process_detects_negative_sentiment():
    response = client.post("/process", json={"text": SAMPLE_TICKET})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "NEGATIVE"
    assert body["sentiment_score"] > 0.9


def test_process_extracts_entities():
    response = client.post("/process", json={"text": SAMPLE_TICKET})
    assert response.status_code == 200
    body = response.json()
    entity_texts = [e["text"] for e in body["entities"]]
    entity_types = [e["type"] for e in body["entities"]]
    # Should find London as LOC and Apple as ORG
    assert any("London" in t for t in entity_texts)
    assert "LOC" in entity_types
    assert "ORG" in entity_types


def test_process_rejects_short_text():
    response = client.post("/process", json={"text": "short"})
    assert response.status_code == 422


def test_batch_process():
    tickets = [SAMPLE_TICKET, "Great service, very happy with my purchase!"]
    response = client.post("/process/batch", json={"texts": tickets})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["results"]) == 2

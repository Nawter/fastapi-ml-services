import pytest
from fastapi.testclient import TestClient
from app.main import app

# TestClient starts the app (including lifespan) for the test session
# Runs in-process — no real server, no ports needed
client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_classify_positive_text():
    response = client.post("/classify", json={"text": "I absolutely love this product!"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "POSITIVE"
    assert body["score"] > 0.9
    assert "inference_ms" in body
    assert "confidence_pct" in body


def test_classify_negative_text():
    response = client.post(
        "/classify", json={"text": "Terrible product, complete waste of money"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "NEGATIVE"
    assert body["score"] > 0.9


def test_classify_rejects_empty_text():
    response = client.post("/classify", json={"text": ""})
    # Pydantic validation fails — FastAPI returns 422 Unprocessable Entity
    assert response.status_code == 422


def test_batch_classify():
    texts = [
        "Great product, very happy",
        "Awful quality, do not buy",
        "It arrived on time",
    ]
    response = client.post("/classify/batch", json={"texts": texts})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["results"]) == 3
    for result in body["results"]:
        assert "label" in result
        assert "score" in result
        assert result["label"] in ("POSITIVE", "NEGATIVE")


def test_detailed_classify_top_k():
    response = client.post(
        "/classify/detailed", json={"text": "It is okay I guess", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    scores = [p["score"] for p in body["predictions"]]
    assert abs(sum(scores) - 1.0) < 0.01  # softmax — should sum to ~1.0

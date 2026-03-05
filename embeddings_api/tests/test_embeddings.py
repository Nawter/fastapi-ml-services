import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["embedding_dim"] == 384


def test_embed_returns_vector():
    response = client.post("/embed", json={"text": "Hello world"})
    assert response.status_code == 200
    body = response.json()
    assert body["dim"] == 384
    assert len(body["embedding"]) == 384
    assert all(isinstance(x, float) for x in body["embedding"])


def test_embed_rejects_empty():
    response = client.post("/embed", json={"text": ""})
    assert response.status_code == 422


def test_similar_sentences_have_high_score():
    response = client.post(
        "/similarity",
        json={
            "text_a": "I love machine learning",
            "text_b": "I enjoy deep learning and AI",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["similarity"] > 0.7
    assert body["interpretation"] in ("Similar", "Very similar")


def test_unrelated_sentences_have_low_score():
    response = client.post(
        "/similarity",
        json={
            "text_a": "The cat sat on the mat",
            "text_b": "Quarterly revenue exceeded expectations",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["similarity"] < 0.5


def test_identical_sentences_score_is_one():
    text = "This is a test sentence"
    response = client.post(
        "/similarity", json={"text_a": text, "text_b": text}
    )
    assert response.status_code == 200
    body = response.json()
    assert abs(body["similarity"] - 1.0) < 0.001

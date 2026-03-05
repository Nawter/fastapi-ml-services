# Embeddings API

Semantic embedding and similarity service using `all-MiniLM-L6-v2` (384-dimensional vectors).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Model status and embedding dimension |
| POST | `/embed` | Embed a single text |
| POST | `/embed/batch` | Embed multiple texts (max 64) |
| POST | `/similarity` | Cosine similarity between two texts |

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

## Run with Docker

```bash
docker compose up --build
```

## Example — similarity

```bash
curl -X POST http://localhost:8001/similarity \
     -H "Content-Type: application/json" \
     -d '{"text_a": "I love machine learning", "text_b": "I enjoy deep learning"}'
```

```json
{
  "text_a": "I love machine learning",
  "text_b": "I enjoy deep learning",
  "similarity": 0.9231,
  "interpretation": "Very similar"
}
```

## Understanding similarity scores

| Score | Meaning |
|-------|---------|
| 0.9+ | Very similar — nearly the same meaning |
| 0.7–0.9 | Similar — clearly related topic |
| 0.5–0.7 | Somewhat related |
| < 0.5 | Not similar |

# Review Classifier API

Sentiment classification service using `distilbert-base-uncased-finetuned-sst-2-english`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Model status |
| POST | `/classify` | Single text classification |
| POST | `/classify/detailed` | Classification with all label scores |
| POST | `/classify/batch` | Batch classification (max 32 texts) |

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for the interactive API.

## Run with Docker

```bash
docker compose up --build
```

Model is cached in `~/.cache/huggingface` — only downloads on first run.

## Run tests

```bash
pytest tests/ -v
```

## Example request

```bash
curl -X POST http://localhost:8000/classify \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is absolutely brilliant!"}'
```

```json
{
  "label": "POSITIVE",
  "score": 0.9998,
  "inference_ms": 34.2,
  "confidence_pct": "99.98%",
  "is_high_confidence": true
}
```

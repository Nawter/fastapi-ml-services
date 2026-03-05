# Document Tagger API

Three-model pipeline for processing support tickets:
- **Summarisation** — `facebook/bart-large-cnn`
- **Sentiment** — `distilbert-base-uncased-finetuned-sst-2-english`
- **NER** — `dbmdz/bert-large-cased-finetuned-conll03-english`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | All 3 model statuses |
| POST | `/process` | Process a single ticket |
| POST | `/process/batch` | Process multiple tickets (max 16) |

## Run locally

```bash
# Pre-download BART first — it's 1.6GB
python -c "from transformers import pipeline; pipeline('summarization', model='facebook/bart-large-cnn')"

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

## Run with Docker

```bash
docker compose up --build
```

> Note: First startup is slow (~2 min) while models load. Subsequent runs load from cache in ~10s.

## Example request

```bash
curl -X POST http://localhost:8002/process \
     -H "Content-Type: application/json" \
     -d '{
       "text": "I ordered a laptop from your London store last Tuesday. The device arrived with a cracked screen. I contacted Apple support but they refused to help. I want a full refund immediately."
     }'
```

```json
{
  "original_text": "...",
  "summary": "Customer received a damaged laptop and wants a refund after Apple support refused to help.",
  "sentiment": "NEGATIVE",
  "sentiment_score": 0.9991,
  "entities": [
    {"text": "London", "type": "LOC", "score": 0.997},
    {"text": "Apple",  "type": "ORG", "score": 0.994}
  ]
}
```

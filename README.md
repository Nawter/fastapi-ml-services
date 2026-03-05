# ML Engineering Interview Prep — Three Production ML APIs

Three fully working ML microservices built with FastAPI, Pydantic, HuggingFace, and Docker.
Each service follows the same production patterns: lifespan model loading, structured logging,
typed schemas, error handling, and a full test suite.

## Services

| Service | Port | Models | Description |
|---------|------|--------|-------------|
| [reviews_api](./reviews_api) | 8000 | DistilBERT SST-2 | Sentiment classification |
| [embeddings_api](./embeddings_api) | 8001 | all-MiniLM-L6-v2 | Semantic embeddings + similarity |
| [tagger_api](./tagger_api) | 8002 | BART + DistilBERT + BERT NER | Summarisation + sentiment + NER |

## Quick start — run all services

```bash
# Pre-download models (only needed once)
python -c "from transformers import pipeline; pipeline('summarization', model='facebook/bart-large-cnn')"

# Start everything
docker compose up --build

# Check all services are healthy
curl http://localhost:8000/health   # classifier
curl http://localhost:8001/health   # embeddings
curl http://localhost:8002/health   # tagger
```

## Project structure

```
.
├── docker-compose.yml          # runs all 3 services together
├── reviews_api/
│   ├── app/
│   │   ├── main.py             # FastAPI app — lifespan, endpoints, middleware
│   │   └── schemas.py          # Pydantic models
│   ├── tests/
│   │   └── test_classifier.py
│   ├── Dockerfile
│   ├── docker-compose.yml      # run this service alone
│   ├── requirements.txt
│   └── README.md
├── embeddings_api/
│   ├── app/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── tests/
│   │   └── test_embeddings.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── README.md
└── tagger_api/
    ├── app/
    │   ├── main.py
    │   └── schemas.py
    ├── tests/
    │   └── test_tagger.py
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md
```

## Key patterns used throughout

### 1. Lifespan model loading
Models are loaded once at startup — not per request. This is the correct pattern for ML services.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = pipeline("text-classification", model="...")
    app.state.model("warm up")   # warm up before accepting traffic
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)
```

### 2. Pydantic schemas with validation

```python
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=1, ge=1, le=2)
```

Invalid inputs automatically return `422 Unprocessable Entity` — no manual validation needed.

### 3. Docker with model cache volume

```yaml
volumes:
  - ~/.cache/huggingface:/cache/huggingface
```

Models download once on first run, then load from local cache. No re-downloading on restart.

### 4. Always `--host 0.0.0.0` in Docker

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Without `0.0.0.0`, uvicorn only listens inside the container and is unreachable from outside.

## Running tests

```bash
# Per service
cd reviews_api && pip install -r requirements.txt && pytest tests/ -v
cd embeddings_api && pip install -r requirements.txt && pytest tests/ -v
cd tagger_api && pip install -r requirements.txt && pytest tests/ -v
```

## Models used

| Model | Size | Task |
|-------|------|------|
| distilbert-base-uncased-finetuned-sst-2-english | ~270MB | Sentiment classification |
| all-MiniLM-L6-v2 | ~80MB | Sentence embeddings |
| dbmdz/bert-large-cased-finetuned-conll03-english | ~1.3GB | Named entity recognition |
| facebook/bart-large-cnn | ~1.6GB | Summarisation |

# fastapi-ml-services

Three production-grade ML microservices built with FastAPI, Pydantic, HuggingFace, and Docker.

Each service follows the same patterns: lifespan model loading, structured logging, typed schemas, error handling, and a full pytest test suite.

---

## Services

| Service | Port | Model(s) | Description |
|---------|------|----------|-------------|
| [reviews_api](./reviews_api) | 8000 | DistilBERT SST-2 | Sentiment classification |
| [embeddings_api](./embeddings_api) | 8001 | all-MiniLM-L6-v2 | Semantic embeddings + cosine similarity |
| [tagger_api](./tagger_api) | 8002 | BART + DistilBERT + BERT NER | Summarisation + sentiment + NER |

---

## Requirements

- Python 3.11+
- Docker Desktop (for Docker runs)
- Git

---

## Option A — Run with Docker (recommended)

No Python setup needed. Docker handles everything.

```bash
# 1. Clone the repo
git clone https://github.com/nawter/fastapi-ml-services.git
cd fastapi-ml-services

# 2. Start all 3 services
docker compose up --build
```

> **First run will be slow** — Docker downloads the models inside the containers (up to 3–4GB total). This only happens once. After that, models are cached and load in seconds.
>
> The tagger_api is the slowest to start — it loads 3 models including BART (1.6GB). Wait for the log line `All models ready` before testing it.

```bash
# 3. Check all services are healthy
curl http://localhost:8000/health   # reviews_api
curl http://localhost:8001/health   # embeddings_api
curl http://localhost:8002/health   # tagger_api
```

To run a single service:

```bash
docker compose up classifier     # reviews_api only
docker compose up embeddings     # embeddings_api only
docker compose up tagger         # tagger_api only
```

---

## Option B — Run locally with a virtual environment

### 1. Clone the repo

```bash
git clone https://github.com/nawter/fastapi-ml-services.git
cd fastapi-ml-services
```

### 2. Create and activate a virtual environment

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

### 3. Install dependencies for the service you want to run

Each service has its own `requirements.txt`. Pick one:

```bash
# reviews_api
cd reviews_api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# embeddings_api
cd embeddings_api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001

# tagger_api
cd tagger_api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

### 4. Deactivate the virtual environment when done

```bash
deactivate
```

---

## Running tests

Make sure you have the venv active and dependencies installed for the service you want to test.

```bash
# reviews_api
cd reviews_api
pip install -r requirements.txt
pytest tests/ -v

# embeddings_api
cd embeddings_api
pip install -r requirements.txt
pytest tests/ -v

# tagger_api
cd tagger_api
pip install -r requirements.txt
pytest tests/ -v
```

---

## Project structure

```
fastapi-ml-services/
├── docker-compose.yml          # runs all 3 services together
│
├── reviews_api/
│   ├── app/
│   │   ├── main.py             # FastAPI app — lifespan, endpoints, middleware
│   │   └── schemas.py          # Pydantic request/response models
│   ├── tests/
│   │   └── test_classifier.py
│   ├── Dockerfile
│   ├── docker-compose.yml      # run this service in isolation
│   ├── requirements.txt
│   └── README.md
│
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
│
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

---

## Key patterns

### Lifespan model loading

Models are loaded once at startup — not per request. Loading per request adds 2–10 seconds of initialisation to every API call.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = pipeline("text-classification", model="...")
    app.state.model("warm up")   # absorb the slow first inference
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)
```

### Pydantic validation

```python
class ClassifyRequest(BaseModel):
    text:  str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=1, ge=1, le=2)
```

Invalid inputs automatically return `422 Unprocessable Entity`. No manual validation needed.

### Docker model cache

```yaml
volumes:
  - ~/.cache/huggingface:/cache/huggingface
```

Models download once on first run and load from disk on every restart. No re-downloading.

### Always `--host 0.0.0.0` in Docker

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Without `0.0.0.0`, uvicorn only listens inside the container and is unreachable from outside.

---

## Models used

| Model | Size | Task |
|-------|------|------|
| distilbert-base-uncased-finetuned-sst-2-english | ~270MB | Sentiment classification |
| all-MiniLM-L6-v2 | ~80MB | Sentence embeddings |
| dbmdz/bert-large-cased-finetuned-conll03-english | ~1.3GB | Named entity recognition |
| facebook/bart-large-cnn | ~1.6GB | Summarisation |

---

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
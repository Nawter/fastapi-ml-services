from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from loguru import logger
import time

from app.schemas import (
    ProcessRequest,
    BatchProcessRequest,
    ProcessResult,
    EntityResult,
)


# ── Lifespan: load all 3 models once at startup ───────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading 3 models — this may take a moment on first run")

    # Load summariser manually — avoids pipeline task registry issue
    app.state.sum_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-cnn"
    )
    app.state.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/bart-large-cnn"
    )

    app.state.sentiment = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0,
    )
    app.state.ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=0,
    )
    app.state.sentiment("warm up")
    app.state.ner("warm up")
    logger.info("All models ready")
    yield
    app.state.sum_tokenizer = app.state.sum_model = None
    app.state.sentiment = app.state.ner = None

app = FastAPI(
    title="Document Tagger",
    version="1.0.0",
    description="Summarisation + Sentiment + NER pipeline for support tickets",
    lifespan=lifespan,
)


# ── Middleware & error handlers ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={(time.time() - t0) * 1000:.1f}ms"
    )
    return response


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )


# ── Helper ─────────────────────────────────────────────────────────────────────
def _summarise(text: str, app) -> str:
    inputs = app.state.sum_tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True
    )
    ids = app.state.sum_model.generate(
        inputs["input_ids"], max_length=60, min_length=15, do_sample=False
    )
    return app.state.sum_tokenizer.decode(ids[0], skip_special_tokens=True)

def _process_one(text: str, app) -> ProcessResult:
    """Run all three models on a single text. Shared by both endpoints."""

    # Summarise — BART needs explicit length params
    summary = _summarise(text, app)

    # Sentiment — truncate to 512 tokens (model limit)
    sent_raw = app.state.sentiment(text[:512])
    sentiment = sent_raw[0]["label"]
    sentiment_score = round(sent_raw[0]["score"], 4)

    # NER — aggregation_strategy='simple' merges token fragments into whole words
    ner_raw = app.state.ner(text[:512])
    entities = [
        EntityResult(
            text=e["word"],
            type=e["entity_group"],
            score=round(e["score"], 4),
        )
        for e in ner_raw
    ]

    return ProcessResult(
        original_text=text,
        summary=summary,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        entities=entities,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    all_loaded = all([
        app.state.sum_model,
        app.state.sentiment,
        app.state.ner
    ])
    return {
        "status": "ok" if all_loaded else "degraded",
        "models": {
            "summariser": "facebook/bart-large-cnn",
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
            "ner": "dbmdz/bert-large-cased-finetuned-conll03-english",
        },
        "all_loaded": all_loaded,
    }


@app.post("/process", response_model=ProcessResult)
def process(request: ProcessRequest):
    if not all([app.state.sum_model, app.state.sentiment, app.state.ner]):
        raise HTTPException(503, "Models not ready")
    return _process_one(request.text, app)


@app.post("/process/batch")
def process_batch(request: BatchProcessRequest):
    if not all([app.state.sum_model, app.state.sentiment, app.state.ner]):
        raise HTTPException(503, "Models not ready")
    results = [_process_one(t, app) for t in request.texts]
    return {"results": [r.model_dump() for r in results], "total": len(results)}

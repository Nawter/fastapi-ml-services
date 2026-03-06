from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from loguru import logger
import time

from app.schemas import (
    ClassifyRequest,
    BatchClassifyRequest,
    ClassifyResult,
    BatchClassifyResult,
    ClassifyResultTopK,
    LabelScore,
)


# ── Lifespan: load model ONCE at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model")
    t0 = time.time()

    try:
        app.state.classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0,  # -1 = CPU, 0 = first GPU
        )
    except Exception as e:
        logger.error(f"FATAL: Failed to load model: {e}")
        raise

    # Smoke test — verify model works before accepting traffic
    try:
        test_result = app.state.classifier("smoke test")
        assert len(test_result) > 0
        assert "label" in test_result[0]
        logger.info(f"Smoke test passed: {test_result[0]}")
    except Exception as e:
        logger.error(f"FATAL: Smoke test failed: {e}")
        raise

    elapsed = (time.time() - t0) * 1000
    logger.info(f"Model loaded and warmed up in {elapsed:.0f}ms")

    yield  # app runs here

    logger.info("Shutting down")
    app.state.classifier = None


app = FastAPI(
    title="Review Classifier",
    version="2.0.0",
    description="Real HuggingFace sentiment classifier — distilbert-base-uncased-finetuned-sst-2-english",
    lifespan=lifespan,
)


# ── Request logging middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={elapsed_ms:.1f}ms"
    )
    return response


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} — {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "model_loaded": app.state.classifier is not None,
    }


@app.post("/classify", response_model=ClassifyResult)
def classify(request: ClassifyRequest):
    if not app.state.classifier:
        raise HTTPException(503, "Model not ready")

    t0 = time.time()
    raw = app.state.classifier(request.text)
    # raw = [{'label': 'POSITIVE', 'score': 0.9998}]
    inference_ms = round((time.time() - t0) * 1000, 1)

    result = ClassifyResult(
        label=raw[0]["label"],
        score=round(raw[0]["score"], 4),
        inference_ms=inference_ms,
    )
    logger.debug(f"Classified: label={result.label} score={result.score} ms={inference_ms}")
    return result


@app.post("/classify/detailed", response_model=ClassifyResultTopK)
def classify_detailed(request: ClassifyRequest):
    if not app.state.classifier:
        raise HTTPException(503, "Model not ready")

    raw = app.state.classifier(request.text, top_k=request.top_k)

    return ClassifyResultTopK(
        text=request.text,
        predictions=[
            LabelScore(label=p["label"], score=round(p["score"], 4)) for p in raw
        ],
    )


@app.post("/classify/batch", response_model=BatchClassifyResult)
def classify_batch(request: BatchClassifyRequest):
    if not app.state.classifier:
        raise HTTPException(503, "Model not ready")

    # Pass the whole list at once — HuggingFace handles batching internally
    raw_results = app.state.classifier(request.texts)

    results = [
        ClassifyResult(label=r["label"], score=round(r["score"], 4))
        for r in raw_results
    ]
    return BatchClassifyResult(results=results, total=len(results))

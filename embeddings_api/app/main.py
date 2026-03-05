from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np
import time

from app.schemas import (
    EmbedRequest,
    EmbedBatchRequest,
    EmbedResult,
    SimilarityRequest,
    SimilarityResult,
)


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norms = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norms == 0:
        return 0.0
    return float(dot / norms)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading sentence-transformers model...")
    t0 = time.time()

    try:
        # all-MiniLM-L6-v2: small (80MB), fast, excellent quality
        app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model: {e}")
        raise

    # Warm up
    app.state.model.encode(["warm up"])
    dim = app.state.model.get_sentence_embedding_dimension()

    logger.info(f"Model ready in {(time.time() - t0) * 1000:.0f}ms — dim={dim}")
    yield
    app.state.model = None


app = FastAPI(
    title="Embeddings Service",
    version="1.0.0",
    description="Semantic embeddings and similarity using all-MiniLM-L6-v2",
    lifespan=lifespan,
)


# ── Middleware ─────────────────────────────────────────────────────────────────
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
    loaded = app.state.model is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model": "all-MiniLM-L6-v2",
        "model_loaded": loaded,
        "embedding_dim": (
            app.state.model.get_sentence_embedding_dimension() if loaded else None
        ),
    }


@app.post("/embed", response_model=EmbedResult)
def embed(request: EmbedRequest):
    if not app.state.model:
        raise HTTPException(503, "Model not ready")

    embedding = app.state.model.encode(request.text).tolist()
    return EmbedResult(text=request.text, embedding=embedding, dim=len(embedding))


@app.post("/embed/batch")
def embed_batch(request: EmbedBatchRequest):
    if not app.state.model:
        raise HTTPException(503, "Model not ready")

    embeddings = app.state.model.encode(request.texts)  # returns np.ndarray
    return {
        "texts": request.texts,
        "embeddings": embeddings.tolist(),  # numpy → list for JSON serialisation
        "dim": embeddings.shape[1],
        "count": len(request.texts),
    }


@app.post("/similarity", response_model=SimilarityResult)
def similarity(request: SimilarityRequest):
    if not app.state.model:
        raise HTTPException(503, "Model not ready")

    emb_a, emb_b = app.state.model.encode([request.text_a, request.text_b])
    score = cosine_similarity(emb_a.tolist(), emb_b.tolist())

    return SimilarityResult(
        text_a=request.text_a,
        text_b=request.text_b,
        similarity=round(score, 4),
    )

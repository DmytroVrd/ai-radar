from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import HealthResponse, IndexResponse, QueryRequest, QueryResponse
from src.config import get_settings
from src.ingestion.fetcher import fetch_all_sources
from src.ingestion.indexer import collection_stats, index_articles, qdrant_is_available
from src.ingestion.scheduler import build_scheduler
from src.retrieval.pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.pipeline = None
    app.state.scheduler = None

    if settings.scheduler_enabled:
        scheduler = build_scheduler(settings)
        scheduler.start()
        app.state.scheduler = scheduler

    yield

    scheduler = getattr(app.state, "scheduler", None)
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(title="AI Radar", lifespan=lifespan)


def _pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        pipeline = RAGPipeline(request.app.state.settings)
        request.app.state.pipeline = pipeline
    return pipeline


@app.get("/")
async def root(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    return {
        "name": "AI Radar",
        "status": "ok" if qdrant_is_available(settings=settings) else "degraded",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "index": "/index",
            "query": "/query",
        },
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    try:
        result = await _pipeline(request).ask(payload.question, top_k=payload.top_k)
        return QueryResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/index", response_model=IndexResponse)
async def index_now(request: Request) -> IndexResponse:
    settings = request.app.state.settings
    if not qdrant_is_available(settings=settings):
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant is unavailable at {settings.qdrant_url}. Start it first.",
        )
    try:
        articles = await fetch_all_sources(settings=settings)
        report = index_articles(articles, settings=settings)
        return IndexResponse(**report.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
async def stats(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    if not qdrant_is_available(settings=settings):
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant is unavailable at {settings.qdrant_url}. Start it first.",
        )
    try:
        return collection_stats(settings=settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    if qdrant_is_available(settings=settings):
        return HealthResponse(status="ok")
    return HealthResponse(status="degraded")

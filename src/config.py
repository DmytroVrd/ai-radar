from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_site_url: str | None
    openrouter_app_name: str | None
    telegram_bot_token: str | None
    qdrant_url: str
    qdrant_collection: str
    chat_model: str
    embedding_model: str
    embedding_dimensions: int | None
    rag_api_url: str
    scheduler_enabled: bool
    scheduler_interval_hours: int
    fetch_limit_hn: int
    fetch_limit_devto: int
    fetch_limit_arxiv: int
    fetch_limit_rss: int
    default_top_k: int
    vector_search_k: int
    bm25_search_k: int
    rerank_top_k: int
    chunk_size: int
    chunk_overlap: int
    request_timeout: float


def _compact_headers(headers: dict[str, str | None]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if value}


def openrouter_headers(settings: Settings) -> dict[str, str]:
    return _compact_headers(
        {
            "HTTP-Referer": settings.openrouter_site_url,
            "X-OpenRouter-Title": settings.openrouter_app_name,
        }
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw_embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_site_url=os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", "AI Radar"),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "ai_articles"),
        chat_model=os.getenv("CHAT_MODEL", "openrouter/free"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        ),
        embedding_dimensions=int(raw_embedding_dimensions) if raw_embedding_dimensions else None,
        rag_api_url=os.getenv("RAG_API_URL", "http://localhost:8000"),
        scheduler_enabled=_bool_env("SCHEDULER_ENABLED", True),
        scheduler_interval_hours=int(os.getenv("SCHEDULER_INTERVAL_HOURS", "6")),
        fetch_limit_hn=int(os.getenv("FETCH_LIMIT_HN", "30")),
        fetch_limit_devto=int(os.getenv("FETCH_LIMIT_DEVTO", "20")),
        fetch_limit_arxiv=int(os.getenv("FETCH_LIMIT_ARXIV", "20")),
        fetch_limit_rss=int(os.getenv("FETCH_LIMIT_RSS", "8")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        vector_search_k=int(os.getenv("VECTOR_SEARCH_K", "10")),
        bm25_search_k=int(os.getenv("BM25_SEARCH_K", "10")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT", "20")),
    )

from __future__ import annotations

from functools import lru_cache

import httpx
from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings, openrouter_headers


class OpenRouterEmbeddings(Embeddings):
    def __init__(self, settings: Settings | None = None, batch_size: int = 32) -> None:
        self.settings = settings or get_settings()
        self.batch_size = batch_size
        self.base_url = self.settings.openrouter_base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            **openrouter_headers(self.settings),
        }

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json={
                "model": self.settings.embedding_model,
                "input": texts,
            },
            timeout=self.settings.request_timeout * 3,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not data:
            raise ValueError("No embedding data received")
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text if text.strip() else " " for text in texts]
        vectors: list[list[float]] = []
        for index in range(0, len(clean_texts), self.batch_size):
            batch = clean_texts[index : index + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(settings: Settings | None = None) -> OpenRouterEmbeddings:
    return OpenRouterEmbeddings(settings=settings)


@lru_cache(maxsize=8)
def infer_embedding_dimensions(
    embedding_model: str,
    api_key: str | None,
    base_url: str,
    site_url: str | None,
    app_name: str | None,
) -> int:
    probe_settings = Settings(
        openrouter_api_key=api_key,
        openrouter_base_url=base_url,
        openrouter_site_url=site_url,
        openrouter_app_name=app_name,
        telegram_bot_token=None,
        qdrant_url="",
        qdrant_collection="",
        chat_model="",
        chat_fallback_models=(),
        embedding_model=embedding_model,
        embedding_dimensions=None,
        rag_api_url="",
        scheduler_enabled=False,
        scheduler_interval_hours=0,
        fetch_limit_hn=0,
        fetch_limit_devto=0,
        fetch_limit_arxiv=0,
        fetch_limit_rss=0,
        default_top_k=0,
        vector_search_k=0,
        bm25_search_k=0,
        rerank_top_k=0,
        chunk_size=0,
        chunk_overlap=0,
        request_timeout=20,
        ask_timeout=180,
    )
    embeddings = build_embeddings(probe_settings)
    return len(embeddings.embed_query("dimension probe"))


def get_embedding_dimensions(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    if settings.embedding_dimensions is not None:
        return settings.embedding_dimensions
    return infer_embedding_dimensions(
        embedding_model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        site_url=settings.openrouter_site_url,
        app_name=settings.openrouter_app_name,
    )

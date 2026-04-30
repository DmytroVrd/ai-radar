from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from src.config import Settings, get_settings
from src.embeddings import build_embeddings
from src.ingestion.indexer import get_qdrant_client, load_all_documents


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _document_key(document: Document) -> str:
    url = document.metadata.get("url", "")
    title = document.metadata.get("title", "")
    chunk_index = document.metadata.get("chunk_index", 0)
    return f"{url}|{title}|{chunk_index}"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Document]],
    limit: int,
    fusion_constant: int = 60,
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}

    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            key = _document_key(document)
            documents[key] = document
            scores[key] += 1.0 / (fusion_constant + rank)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [documents[key] for key, _ in ordered[:limit]]


def bm25_search(query: str, documents: Sequence[Document], limit: int) -> list[Document]:
    if not documents:
        return []

    corpus = [_tokenize(document.page_content) for document in documents]
    if not any(corpus):
        return []

    model = BM25Okapi(corpus)
    scores = model.get_scores(_tokenize(query))
    ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
    return [document for _, document in ranked[:limit]]


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_k: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[Document]:
    docs = list(documents)
    if not docs:
        return []

    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(model_name)
        pairs = [(query, document.page_content) for document in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
        return [document for _, document in ranked[:top_k]]
    except Exception:
        return docs[:top_k]


class HybridRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client: QdrantClient = get_qdrant_client(self.settings)
        try:
            collection_exists = self.client.collection_exists(self.settings.qdrant_collection)
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant is unavailable at {self.settings.qdrant_url}. Start Qdrant and try again."
            ) from exc
        if not collection_exists:
            raise RuntimeError(
                f"Qdrant collection '{self.settings.qdrant_collection}' was not found. Run POST /index first."
            )
        self.embeddings = build_embeddings(self.settings)
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.qdrant_collection,
            embedding=self.embeddings,
        )

    def _vector_search(self, query: str, limit: int) -> list[Document]:
        return self.store.similarity_search(query, k=limit)

    def _bm25_candidates(self, query: str, limit: int) -> list[Document]:
        documents = load_all_documents(client=self.client, settings=self.settings)
        return bm25_search(query, documents, limit=limit)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        top_k = top_k or self.settings.default_top_k
        bm25_docs = self._bm25_candidates(query, limit=self.settings.bm25_search_k)
        vector_docs = self._vector_search(query, limit=self.settings.vector_search_k)
        fused = reciprocal_rank_fusion([bm25_docs, vector_docs], limit=max(top_k, self.settings.rerank_top_k))
        return rerank_documents(query, fused, top_k=top_k)


def unique_sources(documents: Iterable[Document]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, str]] = []
    for document in documents:
        title = str(document.metadata.get("title", "Unknown"))
        url = str(document.metadata.get("url", ""))
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"title": title, "url": url})
    return sources

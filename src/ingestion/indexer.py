from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from src.config import Settings, get_settings
from src.embeddings import build_embeddings, get_embedding_dimensions
from src.models import Article


@dataclass(slots=True)
class IndexingReport:
    article_count: int
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    return QdrantClient(url=settings.qdrant_url, check_compatibility=False)


def qdrant_is_available(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    client = get_qdrant_client(settings)
    try:
        client.get_collections()
        return True
    except Exception:
        return False


def ensure_collection(client: QdrantClient, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if client.collection_exists(settings.qdrant_collection):
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=get_embedding_dimensions(settings),
            distance=models.Distance.COSINE,
        ),
    )


def stable_chunk_id(article_url: str, chunk_index: int, chunk_text: str) -> str:
    raw = f"{article_url}|{chunk_index}|{chunk_text[:120]}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def build_documents(
    articles: list[Article], settings: Settings | None = None
) -> tuple[list[Document], list[str]]:
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    documents: list[Document] = []
    ids: list[str] = []
    for article in articles:
        chunks = splitter.split_text(article.content)
        for chunk_index, chunk in enumerate(chunks):
            metadata = article.as_metadata() | {"chunk_index": chunk_index}
            documents.append(Document(page_content=chunk, metadata=metadata))
            ids.append(stable_chunk_id(article.url, chunk_index, chunk))
    return documents, ids


def index_articles(
    articles: list[Article], settings: Settings | None = None
) -> IndexingReport:
    settings = settings or get_settings()
    client = get_qdrant_client(settings)
    ensure_collection(client, settings)

    documents, ids = build_documents(articles, settings)
    if not documents:
        return IndexingReport(article_count=0, chunk_count=0)

    embeddings = build_embeddings(settings)
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
    store.add_documents(documents=documents, ids=ids)
    return IndexingReport(article_count=len(articles), chunk_count=len(documents))


def load_all_documents(
    client: QdrantClient | None = None,
    settings: Settings | None = None,
    batch_size: int = 256,
) -> list[Document]:
    settings = settings or get_settings()
    client = client or get_qdrant_client(settings)

    if not client.collection_exists(settings.qdrant_collection):
        return []

    documents: list[Document] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=True,
            with_vectors=False,
            limit=batch_size,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            if page_content:
                documents.append(Document(page_content=page_content, metadata=metadata))
        if offset is None:
            break
    return documents


def collection_stats(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    client = get_qdrant_client(settings)

    if not client.collection_exists(settings.qdrant_collection):
        return {
            "collection": settings.qdrant_collection,
            "points_count": 0,
            "documents_loaded": 0,
        }

    info = client.get_collection(settings.qdrant_collection)
    documents = load_all_documents(client=client, settings=settings)
    return {
        "collection": settings.qdrant_collection,
        "points_count": int(getattr(info, "points_count", 0) or 0),
        "documents_loaded": len(documents),
    }

from langchain_core.documents import Document

from src.retrieval.retriever import bm25_search, reciprocal_rank_fusion


def test_bm25_search_returns_keyword_match_first() -> None:
    documents = [
        Document(page_content="Hybrid search improves rag retrieval", metadata={"title": "A"}),
        Document(page_content="Computer vision paper", metadata={"title": "B"}),
    ]

    ranked = bm25_search("hybrid search rag", documents, limit=2)

    assert ranked[0].metadata["title"] == "A"


def test_reciprocal_rank_fusion_merges_rankings() -> None:
    doc_a = Document(page_content="A", metadata={"title": "A", "url": "a", "chunk_index": 0})
    doc_b = Document(page_content="B", metadata={"title": "B", "url": "b", "chunk_index": 0})
    doc_c = Document(page_content="C", metadata={"title": "C", "url": "c", "chunk_index": 0})

    fused = reciprocal_rank_fusion([[doc_a, doc_b], [doc_b, doc_c]], limit=3)

    assert fused[0].metadata["title"] == "B"
    assert {doc.metadata["title"] for doc in fused} == {"A", "B", "C"}

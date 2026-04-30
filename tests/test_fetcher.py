from src.ingestion.fetcher import deduplicate_articles
from src.models import Article


def test_deduplicate_articles_prefers_unique_urls() -> None:
    articles = [
        Article(
            url="https://example.com/post",
            title="RAG for production",
            content="A",
            source="dev.to",
            published="2026-01-01",
        ),
        Article(
            url="https://example.com/post",
            title="RAG for production duplicate",
            content="B",
            source="hackernews",
            published="2026-01-02",
        ),
        Article(
            url="https://example.com/other",
            title="Agents and tools",
            content="C",
            source="arxiv",
            published="2026-01-03",
        ),
    ]

    deduped = deduplicate_articles(articles)

    assert len(deduped) == 2
    assert deduped[0].title == "RAG for production"
    assert deduped[1].title == "Agents and tools"


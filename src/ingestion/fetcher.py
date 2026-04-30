from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from src.config import Settings, get_settings
from src.models import Article

AI_KEYWORDS = ("ai", "llm", "gpt", "rag", "agent", "ml", "model", "embedding")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _article_key(article: Article) -> str:
    parsed = urlparse(article.url)
    normalized_url = f"{parsed.netloc}{parsed.path}".strip("/") if article.url else ""
    return normalized_url.lower() or article.title.strip().lower()


def deduplicate_articles(articles: list[Article]) -> list[Article]:
    seen: set[str] = set()
    deduped: list[Article] = []
    for article in articles:
        key = _article_key(article)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


async def extract_article_content(url: str, client: httpx.AsyncClient) -> str:
    if not url:
        return ""
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body
    if root is None:
        return ""

    text = _normalize_whitespace(root.get_text(" ", strip=True))
    return text[:12000]


def _story_matches_ai(title: str, text: str) -> bool:
    haystack = f"{title} {text}".lower()
    return any(keyword in haystack for keyword in AI_KEYWORDS)


async def fetch_hackernews_ai(limit: int = 30, settings: Settings | None = None) -> list[Article]:
    settings = settings or get_settings()
    async with httpx.AsyncClient(
        timeout=settings.request_timeout,
        follow_redirects=True,
    ) as client:
        response = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
        response.raise_for_status()
        story_ids = response.json()[:100]

        story_tasks = [
            client.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
            for story_id in story_ids
        ]
        story_responses = await asyncio.gather(*story_tasks, return_exceptions=True)

        articles: list[Article] = []
        for item in story_responses:
            if isinstance(item, Exception):
                continue
            story = item.json()
            if not story or story.get("type") != "story":
                continue

            title = story.get("title", "")
            text = story.get("text", "") or ""
            if not _story_matches_ai(title, text):
                continue

            url = story.get("url", "")
            article_content = text or await extract_article_content(url, client) or title
            published = datetime.fromtimestamp(story.get("time", 0), tz=timezone.utc).isoformat()
            articles.append(
                Article(
                    url=url,
                    title=title,
                    content=article_content,
                    source="hackernews",
                    published=published,
                    tags=["AI", "HackerNews"],
                    summary=title,
                )
            )
            if len(articles) >= limit:
                break
        return articles


async def fetch_devto_ai(limit: int = 20, settings: Settings | None = None) -> list[Article]:
    settings = settings or get_settings()
    async with httpx.AsyncClient(
        timeout=settings.request_timeout,
        follow_redirects=True,
    ) as client:
        response = await client.get(
            "https://dev.to/api/articles",
            params={"tag": "ai", "per_page": limit},
        )
        response.raise_for_status()
        items = response.json()

    return [
        Article(
            url=item["url"],
            title=item["title"],
            content=item.get("body_markdown") or item.get("description") or item["title"],
            source="dev.to",
            published=item.get("published_at", ""),
            tags=item.get("tag_list", []),
            summary=item.get("description"),
        )
        for item in items
    ]


async def fetch_arxiv_ai(limit: int = 20, settings: Settings | None = None) -> list[Article]:
    settings = settings or get_settings()
    params = {
        "search_query": 'all:"retrieval augmented generation" OR all:rag OR all:"large language model" OR all:agent',
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": limit,
    }

    async with httpx.AsyncClient(
        timeout=settings.request_timeout,
        follow_redirects=True,
    ) as client:
        response = await client.get("https://export.arxiv.org/api/query", params=params)
        response.raise_for_status()

    articles: list[Article] = []
    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", namespace):
        title = _normalize_whitespace(entry.findtext("atom:title", default="", namespaces=namespace))
        summary = _normalize_whitespace(entry.findtext("atom:summary", default="", namespaces=namespace))
        url = entry.findtext("atom:id", default="", namespaces=namespace).strip()
        published = entry.findtext("atom:published", default="", namespaces=namespace).strip()
        tags = [element.attrib.get("term", "") for element in entry.findall("atom:category", namespace)]
        articles.append(
            Article(
                url=url,
                title=title,
                content=summary or title,
                source="arxiv",
                published=published,
                tags=tags,
                summary=summary[:400],
            )
        )
    return articles


async def fetch_all_sources(settings: Settings | None = None) -> list[Article]:
    settings = settings or get_settings()
    batches = await asyncio.gather(
        fetch_hackernews_ai(limit=settings.fetch_limit_hn, settings=settings),
        fetch_devto_ai(limit=settings.fetch_limit_devto, settings=settings),
        fetch_arxiv_ai(limit=settings.fetch_limit_arxiv, settings=settings),
    )
    articles = [article for batch in batches for article in batch if article.content]
    return deduplicate_articles(articles)

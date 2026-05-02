from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from html import unescape
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from src.config import Settings, get_settings
from src.models import Article

AI_KEYWORDS = (
    "ai",
    "artificial intelligence",
    "llm",
    "large language model",
    "gpt",
    "rag",
    "retrieval",
    "agent",
    "agents",
    "ml",
    "machine learning",
    "model",
    "models",
    "embedding",
    "inference",
    "fine-tuning",
    "eval",
    "open source",
    "tooling",
)

DEVTO_TAGS = ("ai", "machinelearning", "llm", "python", "opensource")

RSS_FEEDS = {
    "huggingface": "https://huggingface.co/blog/feed.xml",
    "openai": "https://openai.com/news/rss.xml",
    "google-ai": "https://blog.google/technology/ai/rss/",
    "simon-willison": "https://simonwillison.net/atom/everything/",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    soup = BeautifulSoup(unescape(text), "html.parser")
    return _normalize_whitespace(soup.get_text(" ", strip=True))


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
        tasks = [
            client.get(
                "https://dev.to/api/articles",
                params={"tag": tag, "per_page": max(1, limit // len(DEVTO_TAGS))},
            )
            for tag in DEVTO_TAGS
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    items = []
    for response in responses:
        if isinstance(response, Exception):
            continue
        response.raise_for_status()
        items.extend(response.json())

    articles = [
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
    return deduplicate_articles(articles)[:limit]


async def fetch_arxiv_ai(limit: int = 20, settings: Settings | None = None) -> list[Article]:
    settings = settings or get_settings()
    params = {
        "search_query": (
            'all:"retrieval augmented generation" OR all:rag OR all:"large language model" '
            'OR all:agent OR all:"open source" OR all:inference OR all:embedding '
            'OR all:"AI tooling" OR all:"machine learning systems"'
        ),
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


def _xml_text(element: ET.Element | None) -> str:
    return _normalize_whitespace(element.text or "") if element is not None else ""


def _first_child(element: ET.Element, names: tuple[str, ...]) -> ET.Element | None:
    for child in element.iter():
        local_name = child.tag.rsplit("}", 1)[-1].lower()
        if local_name in names:
            return child
    return None


def _entry_link(entry: ET.Element) -> str:
    for link in entry.iter():
        local_name = link.tag.rsplit("}", 1)[-1].lower()
        if local_name != "link":
            continue
        href = link.attrib.get("href")
        rel = link.attrib.get("rel", "alternate")
        if href and rel == "alternate":
            return href.strip()
        if link.text:
            return link.text.strip()
    return ""


async def fetch_rss_feed(
    source: str,
    url: str,
    limit: int,
    settings: Settings | None = None,
) -> list[Article]:
    settings = settings or get_settings()
    async with httpx.AsyncClient(
        timeout=settings.request_timeout,
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    root = ET.fromstring(response.text)
    entries = [
        element
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1].lower() in {"item", "entry"}
    ]

    articles: list[Article] = []
    for entry in entries:
        title = _xml_text(_first_child(entry, ("title",)))
        raw_summary = _xml_text(_first_child(entry, ("encoded", "content", "summary", "description")))
        summary = _strip_html(raw_summary)
        content = summary or title
        if not title or not _story_matches_ai(title, content):
            continue

        articles.append(
            Article(
                url=_entry_link(entry),
                title=title,
                content=content,
                source=source,
                published=_xml_text(_first_child(entry, ("published", "updated", "pubdate"))),
                tags=["AI", "LLM", "AI engineering", source],
                summary=summary[:400],
            )
        )
        if len(articles) >= limit:
            break
    return articles


async def fetch_ai_engineering_feeds(settings: Settings | None = None) -> list[Article]:
    settings = settings or get_settings()
    batches = await asyncio.gather(
        *[
            fetch_rss_feed(
                source=source,
                url=url,
                limit=settings.fetch_limit_rss,
                settings=settings,
            )
            for source, url in RSS_FEEDS.items()
        ],
        return_exceptions=True,
    )

    articles: list[Article] = []
    for batch in batches:
        if isinstance(batch, Exception):
            continue
        articles.extend(batch)
    return deduplicate_articles(articles)


async def fetch_all_sources(settings: Settings | None = None) -> list[Article]:
    settings = settings or get_settings()
    batches = await asyncio.gather(
        fetch_hackernews_ai(limit=settings.fetch_limit_hn, settings=settings),
        fetch_devto_ai(limit=settings.fetch_limit_devto, settings=settings),
        fetch_arxiv_ai(limit=settings.fetch_limit_arxiv, settings=settings),
        fetch_ai_engineering_feeds(settings=settings),
    )
    articles = [article for batch in batches for article in batch if article.content]
    return deduplicate_articles(articles)

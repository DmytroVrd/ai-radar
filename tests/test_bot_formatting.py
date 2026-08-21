from __future__ import annotations

import httpx

from src.bot.handlers import (
    TELEGRAM_MESSAGE_LIMIT,
    _answer_without_generated_sources,
    _describe_http_error,
    _query_timeout,
    _render_answer_messages,
)
from src.config import get_settings


def test_answer_removes_only_generated_sources() -> None:
    answer = """AI coding agents can:

* **Plan and write code** — with tools [1].
* **Run tests** — and inspect failures [2].

Sources:
1. [Wrong source](https://example.invalid)
"""

    cleaned = _answer_without_generated_sources(answer)

    assert "* **Plan and write code** — with tools [1]." in cleaned
    assert "* **Run tests** — and inspect failures [2]." in cleaned
    assert "Wrong source" not in cleaned
    assert "example.invalid" not in cleaned


def test_render_answer_uses_numbered_escaped_clickable_sources() -> None:
    rendered = _render_answer_messages(
        "• A short answer [1].",
        [
            {"title": "RAG & Agents", "url": "https://example.com/one?a=1&b=2"},
            {"title": "Second <source>", "url": "https://example.com/two"},
        ],
    )

    assert len(rendered) == 1
    message = rendered[0]
    assert "<b>Sources</b>" in message
    assert '1. <a href="https://example.com/one?a=1&amp;b=2">RAG &amp; Agents</a>' in message
    assert '2. <a href="https://example.com/two">Second &lt;source&gt;</a>' in message
    assert "**" not in message


def test_render_answer_rejects_unsafe_source_url_and_limits_count() -> None:
    sources = [
        {"title": "Unsafe", "url": "javascript:alert(1)"},
        {"title": "Two", "url": "https://example.com/2"},
        {"title": "Three", "url": "https://example.com/3"},
        {"title": "Four", "url": "https://example.com/4"},
    ]

    rendered = "\n".join(_render_answer_messages("Useful answer.", sources))

    assert "1. Unsafe" in rendered
    assert "javascript:" not in rendered
    assert "3. <a" in rendered
    assert "Four" not in rendered


def test_render_answer_splits_without_truncating_or_breaking_markup() -> None:
    answer = "\n".join(f"* **Point {index}** — " + "detail " * 80 for index in range(20))
    rendered = _render_answer_messages(
        answer,
        [{"title": "A source", "url": "https://example.com/article"}],
    )

    assert len(rendered) > 1
    assert all(len(message) <= TELEGRAM_MESSAGE_LIMIT for message in rendered)
    joined = "\n".join(rendered)
    assert "Point 0" in joined
    assert "Point 19" in joined
    assert "**" not in joined
    assert joined.count("<b>") == joined.count("</b>")


def test_exact_long_sample_uses_bullets_bold_and_deterministic_sources() -> None:
    answer = """AI coding agents today can:
* **Plan and write code autonomously** – they can draft, refactor, and test code [2].
* **Execute tasks end-to-end** – from setup to running scripts [1].
* **Interact with APIs and services** – they can integrate external services [4].
In short, modern agents can execute code-centric tasks.
Sources:
1. Wrong model-authored source https://invalid.example
"""
    messages = _render_answer_messages(
        answer,
        [{"title": "Trusted source", "url": "https://example.com/trusted"}],
    )
    rendered = "\n".join(messages)

    assert "• <b>Plan and write code autonomously</b>" in rendered
    assert "• <b>Execute tasks end-to-end</b>" in rendered
    assert "• <b>Interact with APIs and services</b>" in rendered
    assert "Wrong model-authored" not in rendered
    assert '1. <a href="https://example.com/trusted">Trusted source</a>' in rendered
    assert "**" not in rendered


def test_query_timeout_message_does_not_blame_indexing() -> None:
    request = httpx.Request("POST", "http://localhost:8013/query")
    error = httpx.ReadTimeout("timed out", request=request)

    message = _describe_http_error(error, operation="query")

    assert message == "The AI response took too long. Please try the question again."
    assert "index" not in message.lower()


def test_query_timeout_is_independent_from_short_http_timeout(monkeypatch) -> None:
    monkeypatch.setenv("REQUEST_TIMEOUT", "20")
    monkeypatch.setenv("ASK_TIMEOUT", "180")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.request_timeout == 20
    assert _query_timeout(settings) == 180
    get_settings.cache_clear()

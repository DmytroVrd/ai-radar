from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import httpx
from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from src.config import Settings


TELEGRAM_MESSAGE_LIMIT = 4096
SOURCE_LIMIT = 3
MESSAGE_MARGIN = 96


def _answer_without_generated_sources(answer: str) -> str:
    """Remove only the model-authored source appendix; the API owns citations."""
    text = answer.replace("\r\n", "\n").replace("\\*", "*")
    # The API response owns the authoritative source list. Discard any source
    # section improvised by the model so links and numbering cannot diverge.
    text = re.split(r"(?im)^\s*(?:sources|references)\s*:\s*$", text, maxsplit=1)[0]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or "The indexed sources do not contain enough information to answer this question."


def _plain_markdown(text: str) -> str:
    """Remove Markdown syntax without removing any answer content."""
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*(?:[-*+]\s*|\d+[.)]\s+)", "• ", text)
    return text.replace("*", "").replace("__", "")


def _format_markdown_line(line: str) -> str:
    """Render the small Markdown subset produced by chat models as safe HTML."""
    line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
    line = re.sub(r"^\s*(?:[-*+]\s*|\d+[.)]\s+)", "• ", line)
    escaped = html.escape(line)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__([^_]+)__", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    # Never expose unmatched Markdown controls in Telegram.
    return escaped.replace("*", "").replace("__", "")


def _split_plain_text(text: str, limit: int) -> list[str]:
    """Split an oversized line into independently escaped, valid HTML pieces."""
    chunks: list[str] = []
    remaining = _plain_markdown(text).strip()
    while remaining:
        candidate = remaining[:limit]
        if len(remaining) > limit:
            boundary = max(candidate.rfind(" "), candidate.rfind(". "))
            if boundary > limit // 2:
                candidate = candidate[:boundary]
        candidate = candidate.strip()
        if not candidate:
            candidate = remaining[:limit]
        chunks.append(html.escape(candidate))
        remaining = remaining[len(candidate) :].lstrip()
    return chunks


def _answer_chunks(answer: str) -> list[str]:
    """Create complete HTML messages without truncating the model's answer."""
    limit = TELEGRAM_MESSAGE_LIMIT - MESSAGE_MARGIN
    blocks: list[str] = []
    for line in _answer_without_generated_sources(answer).splitlines():
        if not line.strip():
            blocks.append("")
            continue
        rendered = _format_markdown_line(line.strip())
        if len(rendered) <= limit:
            blocks.append(rendered)
        else:
            blocks.extend(_split_plain_text(line, limit))

    messages: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n{block}".strip() if current else block
        if current and len(candidate) > limit:
            messages.append(current.rstrip())
            current = block
        else:
            current = candidate
    if current:
        messages.append(current.rstrip())
    return messages


def _safe_url(value: object) -> str | None:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return url


def _render_sources(sources: list[dict[str, str]]) -> str:
    if not sources:
        return "<b>Sources</b>\nNo sources returned."
    lines = ["<b>Sources</b>"]
    for index, source in enumerate(sources[:SOURCE_LIMIT], start=1):
        title = html.escape(str(source.get("title") or "Untitled source").strip()[:140])
        url = _safe_url(source.get("url"))
        if url:
            lines.append(f'{index}. <a href="{html.escape(url, quote=True)}">{title}</a>')
        else:
            lines.append(f"{index}. {title}")
    return "\n".join(lines)


def _render_answer_messages(answer: str, sources: list[dict[str, str]]) -> list[str]:
    source_block = _render_sources(sources)
    messages = _answer_chunks(answer)
    if messages and len(messages[-1]) + len(source_block) + 2 <= TELEGRAM_MESSAGE_LIMIT:
        messages[-1] = f"{messages[-1]}\n\n{source_block}"
    else:
        messages.append(source_block)
    return messages


def _is_admin(message: Message, settings: Settings) -> bool:
    if not message.from_user:
        return False
    return message.from_user.id in settings.telegram_admin_ids


def _describe_http_error(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        return f"{response.status_code}: {detail or response.reason_phrase}"
    if isinstance(exc, httpx.ConnectError):
        return "API is not reachable. Start FastAPI first with python main.py."
    if isinstance(exc, httpx.ReadTimeout):
        return "Request timed out. Indexing can take a few minutes; try again or increase REQUEST_TIMEOUT."
    return str(exc) or exc.__class__.__name__


def build_router(settings: Settings) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start_handler(message: Message) -> None:
        admin_hint = ""
        if _is_admin(message, settings):
            admin_hint = "\nAdmin: use /index to pull fresh content."
        await message.answer(
            "AI Radar is ready.\n"
            "Use /ask <question> to query indexed articles.\n"
            "Use /stats to inspect the current index."
            f"{admin_hint}"
        )

    @router.message(Command("ask"))
    async def ask_handler(message: Message) -> None:
        text = message.text or ""
        question = text.replace("/ask", "", 1).strip()
        if not question:
            await message.answer("Example: /ask What changed in RAG over the last month?")
            return

        await message.answer("Searching indexed AI articles...")
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                response = await client.post(
                    f"{settings.rag_api_url.rstrip('/')}/query",
                    json={"question": question, "top_k": settings.default_top_k},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            await message.answer(f"Request failed: {_describe_http_error(exc)}")
            return

        rendered_messages = _render_answer_messages(
            str(data.get("answer", "")), data.get("sources", [])
        )
        for rendered in rendered_messages:
            await message.answer(rendered, parse_mode="HTML", disable_web_page_preview=True)

    @router.message(Command("index"))
    async def index_handler(message: Message) -> None:
        if not settings.telegram_admin_ids:
            await message.answer("Admin commands are disabled. Set TELEGRAM_ADMIN_IDS first.")
            return
        if not _is_admin(message, settings):
            await message.answer("This command is admin-only.")
            return

        await message.answer("Indexing the latest articles...")
        try:
            index_timeout = max(settings.request_timeout, 180)
            async with httpx.AsyncClient(timeout=index_timeout) as client:
                response = await client.post(f"{settings.rag_api_url.rstrip('/')}/index")
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            await message.answer(f"Indexing failed: {_describe_http_error(exc)}")
            return

        await message.answer(
            f"Done. Indexed {data['article_count']} articles into {data['chunk_count']} chunks."
        )

    @router.message(Command("stats"))
    async def stats_handler(message: Message) -> None:
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                response = await client.get(f"{settings.rag_api_url.rstrip('/')}/stats")
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            await message.answer(f"Stats request failed: {_describe_http_error(exc)}")
            return

        await message.answer(
            f"Collection: {data['collection']}\n"
            f"Points: {data['points_count']}\n"
            f"Loaded docs: {data['documents_loaded']}"
        )

    return router

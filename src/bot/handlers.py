from __future__ import annotations

import httpx
from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from src.config import Settings


def _render_sources(sources: list[dict[str, str]]) -> str:
    if not sources:
        return "No sources returned."
    lines = []
    for index, source in enumerate(sources[:3], start=1):
        lines.append(f"{index}. {source['title']}\n{source['url']}")
    return "\n\n".join(lines)


def build_router(settings: Settings) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start_handler(message: Message) -> None:
        await message.answer(
            "AI Radar is ready.\n"
            "Use /ask <question> to query indexed articles.\n"
            "Use /index to pull fresh content."
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
            await message.answer(f"Request failed: {exc}")
            return

        sources = _render_sources(data.get("sources", []))
        await message.answer(f"{data['answer']}\n\nSources:\n{sources}")

    @router.message(Command("index"))
    async def index_handler(message: Message) -> None:
        await message.answer("Indexing the latest articles...")
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                response = await client.post(f"{settings.rag_api_url.rstrip('/')}/index")
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            await message.answer(f"Indexing failed: {exc}")
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
            await message.answer(f"Stats request failed: {exc}")
            return

        await message.answer(
            f"Collection: {data['collection']}\n"
            f"Points: {data['points_count']}\n"
            f"Loaded docs: {data['documents_loaded']}"
        )

    return router


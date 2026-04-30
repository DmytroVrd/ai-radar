from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.config import Settings, get_settings
from src.ingestion.fetcher import fetch_all_sources
from src.ingestion.indexer import index_articles

logger = logging.getLogger(__name__)


async def run_ingestion_cycle(settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    articles = await fetch_all_sources(settings=settings)
    report = index_articles(articles, settings=settings)
    logger.info(
        "Ingestion completed: %s articles, %s chunks",
        report.article_count,
        report.chunk_count,
    )
    return {"articles": report.article_count, "chunks": report.chunk_count}


def build_scheduler(settings: Settings | None = None) -> AsyncIOScheduler:
    settings = settings or get_settings()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_ingestion_cycle,
        trigger=IntervalTrigger(hours=settings.scheduler_interval_hours),
        kwargs={"settings": settings},
        id="ai-radar-ingestion",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    return scheduler


from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher

from src.bot.handlers import build_router
from src.config import get_settings


async def main() -> None:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required to run the bot.")

    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(settings))
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())


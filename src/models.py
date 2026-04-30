from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Article:
    url: str
    title: str
    content: str
    source: str
    published: str
    tags: list[str] = field(default_factory=list)
    summary: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "published": self.published,
            "tags": self.tags,
            "summary": self.summary or "",
        }


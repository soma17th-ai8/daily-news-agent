from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class NewsArticle:
    title: str
    summary: str
    link: str
    source: str
    published_at: str
    keyword: str

    @property
    def id(self) -> str:
        return sha256(self.link.encode("utf-8")).hexdigest()

    def document_text(self) -> str:
        return f"{self.title}\n\n{self.summary}".strip()

    def to_metadata(self) -> dict[str, str]:
        return {
            "title": self.title,
            "summary": self.summary,
            "link": self.link,
            "source": self.source,
            "published_at": self.published_at,
            "keyword": self.keyword,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> "NewsArticle":
        return cls(
            title=metadata.get("title", ""),
            summary=metadata.get("summary", ""),
            link=metadata.get("link", ""),
            source=metadata.get("source", ""),
            published_at=metadata.get("published_at", ""),
            keyword=metadata.get("keyword", ""),
        )


@dataclass(frozen=True)
class ArticleMatch:
    article: NewsArticle
    score: float | None = None


@dataclass(frozen=True)
class BriefingResult:
    interest: str
    keywords: list[str]
    collected_count: int
    stored_count: int
    selected_articles: list[NewsArticle]
    briefing_markdown: str
    errors: list[str]
    skipped_existing_count: int = 0


@dataclass(frozen=True)
class CollectionResult:
    interest: str
    keywords: list[str]
    collected_articles: list[NewsArticle]
    collected_count: int
    stored_count: int
    skipped_existing_count: int
    errors: list[str]

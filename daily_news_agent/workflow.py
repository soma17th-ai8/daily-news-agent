from __future__ import annotations

from daily_news_agent.ai_client import AIClient
from daily_news_agent.models import BriefingResult, NewsArticle
from daily_news_agent.news_source import GoogleNewsRssClient, NewsSourceError
from daily_news_agent.preprocessor import deduplicate_articles, normalize_keywords
from daily_news_agent.vector_store import ChromaArticleStore


class DailyNewsWorkflow:
    def __init__(
        self,
        news_source: GoogleNewsRssClient,
        vector_store: ChromaArticleStore,
        ai_client: AIClient,
    ) -> None:
        self.news_source = news_source
        self.vector_store = vector_store
        self.ai_client = ai_client

    def run(
        self,
        interest: str,
        keyword_text: str,
        per_keyword_limit: int = 10,
        top_k: int = 8,
    ) -> BriefingResult:
        keywords = normalize_keywords(keyword_text, limit=3) or normalize_keywords(interest, limit=3)
        if not keywords:
            raise ValueError("관심 분야 또는 검색 키워드를 입력해야 합니다.")

        collected_articles: list[NewsArticle] = []
        errors: list[str] = []
        for keyword in keywords:
            try:
                collected_articles.extend(self.news_source.fetch(keyword, limit=per_keyword_limit))
            except NewsSourceError as exc:
                errors.append(str(exc))

        deduplicated_articles = deduplicate_articles(collected_articles)
        stored_count = 0
        if deduplicated_articles:
            embeddings = self.ai_client.embed_documents(
                [article.document_text() for article in deduplicated_articles]
            )
            stored_count = self.vector_store.upsert_articles(deduplicated_articles, embeddings)

        selected_articles = self._select_articles(
            interest=interest,
            fallback_articles=deduplicated_articles,
            top_k=top_k,
        )
        briefing = self.ai_client.generate_briefing(interest, selected_articles)

        return BriefingResult(
            interest=interest,
            keywords=keywords,
            collected_count=len(deduplicated_articles),
            stored_count=stored_count,
            selected_articles=selected_articles,
            briefing_markdown=briefing,
            errors=errors,
        )

    def _select_articles(
        self,
        interest: str,
        fallback_articles: list[NewsArticle],
        top_k: int,
    ) -> list[NewsArticle]:
        query_embedding = self.ai_client.embed_query(interest)
        matches = self.vector_store.query(query_embedding=query_embedding, top_k=top_k)
        selected_articles = [match.article for match in matches if match.article.link]
        return selected_articles or fallback_articles[:top_k]


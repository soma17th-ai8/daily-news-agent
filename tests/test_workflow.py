import unittest

from daily_news_agent.models import ArticleMatch, NewsArticle
from daily_news_agent.workflow import DailyNewsWorkflow


class FakeNewsSource:
    def __init__(self, articles_by_keyword):
        self.articles_by_keyword = articles_by_keyword

    def fetch(self, keyword, limit=10):
        return self.articles_by_keyword.get(keyword, [])[:limit]


class CountingAIClient:
    def __init__(self):
        self.embedded_texts = []
        self.queries = []

    def embed_documents(self, texts):
        self.embedded_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [0.0, 1.0]

    def generate_briefing(self, interest, articles):
        return f"{interest}: {len(articles)}건"


class FakeVectorStore:
    def __init__(self, existing_ids=None, matches=None):
        self.existing_ids_value = set(existing_ids or [])
        self.upserted_articles = []
        self.query_keywords = []
        self.matches = matches or []

    def existing_ids(self, article_ids):
        return self.existing_ids_value.intersection(article_ids)

    def upsert_articles(self, articles, embeddings):
        self.upserted_articles.extend(articles)
        return len(articles)

    def query(self, query_embedding, top_k=8, keywords=None):
        self.query_keywords.append(keywords)
        return self.matches[:top_k]


def make_article(title, link, keyword):
    return NewsArticle(
        title=title,
        summary=f"{title} 요약",
        link=link,
        source="Example",
        published_at="2026-05-05",
        keyword=keyword,
    )


class WorkflowTests(unittest.TestCase):
    def test_collect_and_store_skips_existing_articles_before_embedding(self):
        existing = make_article("기존 기사", "https://example.com/existing", "AI")
        fresh = make_article("새 기사", "https://example.com/fresh", "AI")
        ai_client = CountingAIClient()
        vector_store = FakeVectorStore(existing_ids={existing.id})
        workflow = DailyNewsWorkflow(
            news_source=FakeNewsSource({"AI": [existing, fresh]}),
            vector_store=vector_store,
            ai_client=ai_client,
        )

        result = workflow.collect_and_store("AI 산업", "AI", per_keyword_limit=3)

        self.assertEqual(result.collected_count, 2)
        self.assertEqual(result.stored_count, 1)
        self.assertEqual(result.skipped_existing_count, 1)
        self.assertEqual([article.link for article in vector_store.upserted_articles], [fresh.link])
        self.assertEqual(len(ai_client.embedded_texts), 1)
        self.assertIn("새 기사", ai_client.embedded_texts[0])

    def test_generate_briefing_queries_with_keyword_metadata_filter_first(self):
        article = make_article("선별 기사", "https://example.com/selected", "AI")
        ai_client = CountingAIClient()
        vector_store = FakeVectorStore(matches=[ArticleMatch(article=article, score=0.8)])
        workflow = DailyNewsWorkflow(
            news_source=FakeNewsSource({}),
            vector_store=vector_store,
            ai_client=ai_client,
        )

        result = workflow.generate_briefing("AI 산업", ["AI", "반도체"], top_k=3)

        self.assertEqual(vector_store.query_keywords[0], ["AI", "반도체"])
        self.assertEqual([selected.link for selected in result.selected_articles], [article.link])
        self.assertEqual(result.briefing_markdown, "AI 산업: 1건")


if __name__ == "__main__":
    unittest.main()

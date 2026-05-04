from __future__ import annotations

from collections.abc import Sequence

from daily_news_agent.models import ArticleMatch, NewsArticle


class ChromaArticleStore:
    def __init__(self, path: str, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb 패키지가 설치되어 있지 않습니다.") from exc

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_articles(
        self,
        articles: Sequence[NewsArticle],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if not articles:
            return 0
        if len(articles) != len(embeddings):
            raise ValueError("articles와 embeddings 개수가 일치해야 합니다.")

        self.collection.upsert(
            ids=[article.id for article in articles],
            documents=[article.document_text() for article in articles],
            metadatas=[article.to_metadata() for article in articles],
            embeddings=[list(embedding) for embedding in embeddings],
        )
        return len(articles)

    def query(self, query_embedding: Sequence[float], top_k: int = 8) -> list[ArticleMatch]:
        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[ArticleMatch] = []
        for metadata, distance in zip(metadatas, distances, strict=False):
            score = None if distance is None else max(0.0, 1.0 - float(distance))
            matches.append(ArticleMatch(article=NewsArticle.from_metadata(metadata), score=score))
        return matches

    def count(self) -> int:
        return int(self.collection.count())


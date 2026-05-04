from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    upstage_api_key: str
    upstage_base_url: str
    upstage_chat_model: str
    upstage_document_embedding_model: str
    upstage_query_embedding_model: str
    chroma_path: str
    chroma_collection_name: str
    per_keyword_limit: int
    top_k: int
    request_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        api_key = os.getenv("UPSTAGE_API_KEY", "").strip()
        document_model = os.getenv(
            "UPSTAGE_DOCUMENT_EMBEDDING_MODEL",
            "solar-embedding-1-large-passage",
        )
        query_model = os.getenv(
            "UPSTAGE_QUERY_EMBEDDING_MODEL",
            "solar-embedding-1-large-query",
        )
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "").strip()
        if not collection_name:
            model_slug = _slugify(document_model if api_key else "demo-embedding")
            collection_name = f"daily_news_articles_{model_slug}"

        return cls(
            upstage_api_key=api_key,
            upstage_base_url=os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1/solar"),
            upstage_chat_model=os.getenv("UPSTAGE_CHAT_MODEL", "solar-mini"),
            upstage_document_embedding_model=document_model,
            upstage_query_embedding_model=query_model,
            chroma_path=os.getenv("CHROMA_PATH", "data/chroma"),
            chroma_collection_name=collection_name,
            per_keyword_limit=int(os.getenv("PER_KEYWORD_LIMIT", "10")),
            top_k=int(os.getenv("TOP_K", "8")),
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10")),
        )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return slug or "default"


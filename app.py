from __future__ import annotations

import streamlit as st

from daily_news_agent.ai_client import DemoAIClient, UpstageAIClient
from daily_news_agent.config import Settings
from daily_news_agent.news_source import GoogleNewsRssClient
from daily_news_agent.vector_store import ChromaArticleStore
from daily_news_agent.workflow import DailyNewsWorkflow


def create_ai_client(settings: Settings) -> DemoAIClient | UpstageAIClient:
    if not settings.upstage_api_key:
        return DemoAIClient()
    return UpstageAIClient(
        api_key=settings.upstage_api_key,
        base_url=settings.upstage_base_url,
        chat_model=settings.upstage_chat_model,
        document_embedding_model=settings.upstage_document_embedding_model,
        query_embedding_model=settings.upstage_query_embedding_model,
    )


def main() -> None:
    settings = Settings.from_env()

    st.set_page_config(page_title="Daily News Agent", layout="wide")
    st.title("Daily News Agent")
    st.caption("관심 분야를 입력하면 Google News RSS에서 하루치 뉴스를 모아 로컬 Vector DB에 저장하고 브리핑을 생성합니다.")

    with st.sidebar:
        st.subheader("실행 설정")
        st.write(f"AI 모드: {'Upstage' if settings.upstage_api_key else 'Demo'}")
        st.write(f"Vector DB: `{settings.chroma_path}`")
        st.write(f"Collection: `{settings.chroma_collection_name}`")
        per_keyword_limit = st.number_input(
            "키워드별 수집 기사 수",
            min_value=3,
            max_value=20,
            value=settings.per_keyword_limit,
            step=1,
        )
        top_k = st.number_input("요약에 사용할 기사 수", min_value=3, max_value=12, value=settings.top_k, step=1)

    interest = st.text_input("관심 분야", value="AI 산업 동향")
    keyword_text = st.text_input("검색 키워드 3개", value="AI, 반도체, 스타트업")

    if st.button("뉴스 수집 및 브리핑 생성", type="primary"):
        try:
            ai_client = create_ai_client(settings)
            vector_store = ChromaArticleStore(
                path=settings.chroma_path,
                collection_name=settings.chroma_collection_name,
            )
            workflow = DailyNewsWorkflow(
                news_source=GoogleNewsRssClient(timeout_seconds=settings.request_timeout_seconds),
                vector_store=vector_store,
                ai_client=ai_client,
            )

            with st.spinner("뉴스를 수집하고 브리핑을 생성하는 중입니다..."):
                result = workflow.run(
                    interest=interest,
                    keyword_text=keyword_text,
                    per_keyword_limit=int(per_keyword_limit),
                    top_k=int(top_k),
                )

            col1, col2, col3 = st.columns(3)
            col1.metric("검색 키워드", len(result.keywords))
            col2.metric("정제된 기사", result.collected_count)
            col3.metric("Vector DB 저장", result.stored_count)

            if result.errors:
                st.warning("\n".join(result.errors))

            st.subheader("생성된 브리핑")
            st.markdown(result.briefing_markdown)

            st.subheader("선별된 기사")
            for article in result.selected_articles:
                with st.expander(article.title):
                    st.write(article.summary or "요약 없음")
                    st.write(f"출처: {article.source}")
                    st.write(f"발행일: {article.published_at or '알 수 없음'}")
                    st.write(f"키워드: {article.keyword}")
                    st.link_button("원문 열기", article.link)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()

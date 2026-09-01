"""로컬 서재의 논문 목록, 초록 확인 및 초록 번역 화면."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from apps.ui import render_page_heading
from feature.search_list import LocalLibraryBot


ABSTRACT_TRANSLATIONS_KEY = "abstract_translations"


@st.cache_resource(show_spinner=False)
def _get_library_bot() -> LocalLibraryBot:
    return LocalLibraryBot()


def _load_papers() -> list[dict[str, Any]]:
    bot = _get_library_bot()
    paper_ids = bot.get_all_json_ids()
    return bot.fetch_full_data_from_db(paper_ids) if paper_ids else []


def render_paper_list_page() -> None:
    """보유 논문을 선택하면 초록과 번역 기능을 표시한다."""
    render_page_heading(
        "내 논문 리스트",
        "서재에 저장한 논문의 초록을 확인하고 한국어로 번역할 수 있습니다.",
    )
    try:
        papers = _load_papers()
    except Exception as exc:
        st.error(f"논문 목록을 불러오지 못했습니다: {exc}")
        return

    if not papers:
        st.markdown(
            """
            <div class="empty-guide">
                저장된 논문이 없습니다.<br>
                <strong>논문 검색</strong>에서 논문을 먼저 저장해 주세요.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    title_query = st.text_input(
        "보유 논문 검색", placeholder="저장된 논문 제목에서 검색"
    ).strip().lower()
    if title_query:
        papers = [paper for paper in papers if title_query in paper["title"].lower()]
    if not papers:
        st.info("제목과 일치하는 보유 논문이 없습니다.")
        return

    paper_by_id = {paper["id"]: paper for paper in papers}
    selected_id = st.selectbox(
        f"보유 논문 ({len(papers)}개)",
        options=list(paper_by_id),
        format_func=lambda paper_id: paper_by_id[paper_id]["title"],
    )
    paper = paper_by_id[selected_id]
    st.markdown(
        f"""
        <div class="paper-detail-card">
            <h2>{escape(str(paper['title']))}</h2>
            <p>{escape(str(paper.get('authors') or '저자 정보 없음'))}</p>
            <p>arXiv:{escape(str(paper['id']))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 초록")
    st.write(paper.get("summary") or "초록 정보가 없습니다.")

    pdf_column, translate_column = st.columns(2)
    with pdf_column:
        if paper.get("pdf_url"):
            st.link_button("원문 PDF 열기", paper["pdf_url"], use_container_width=True)
    with translate_column:
        translate_clicked = st.button(
            "초록 한국어 번역", type="primary", use_container_width=True
        )

    translations = st.session_state.setdefault(ABSTRACT_TRANSLATIONS_KEY, {})
    if translate_clicked:
        summary = str(paper.get("summary") or "")
        if not summary.strip():
            st.warning("번역할 초록이 없습니다.")
        else:
            with st.spinner("초록을 한국어로 번역하고 있습니다..."):
                translations[paper["id"]] = _get_library_bot().translate_summary(summary)

    translated = translations.get(paper["id"])
    if translated:
        st.markdown("### 한국어 초록")
        with st.container(border=True):
            st.write(translated)

"""Academic Paper RAG Chatbot의 Streamlit 실행 파일.

실행 방법
---------
프로젝트 최상위 폴더에서 아래 명령어를 실행한다.

    pip install -r requirements.txt
    streamlit run web_app.py

실제 화면은 ``apps`` 폴더에 기능별로 모듈화되어 있다. 이 파일은 실행 설정,
사이드바 메뉴, 화면 연결만 담당한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apps.deep_search_app import render_deep_search_page
from apps.paper_list_app import render_paper_list_page
from apps.search_app import render_search_page
from apps.translation_summary_app import render_translation_summary_page
from apps.ui import apply_global_styles, image_data_url


PAGES = {
    "논문 검색": render_search_page,
    "내 서재": render_paper_list_page,
    "논문 번역 및 요약": render_translation_summary_page,
    "딥서치": render_deep_search_page,
}
LOGO_PATH = PROJECT_ROOT / "apps" / "paper_scholar_logo.png"


def main() -> None:
    """사이드바에서 선택한 Streamlit 화면을 실행한다."""
    st.set_page_config(
        page_title="Academic Paper RAG Chatbot",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_global_styles()
    logo_src = image_data_url(str(LOGO_PATH), LOGO_PATH.stat().st_mtime)

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <span class="sidebar-brand-icon">
                    <img src="{logo_src}" alt="Paper Scholar 책 로고">
                </span>
                <div>
                    <strong>Paper Scholar</strong>
                    <small>Academic RAG Assistant</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_page = st.radio(
            "메뉴",
            options=list(PAGES),
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div class="sidebar-help">
                <strong>이용 순서</strong><br>
                검색 → 내 서재 → 번역·요약 → 딥서치
            </div>
            """,
            unsafe_allow_html=True,
        )

    PAGES[selected_page]()


if __name__ == "__main__":
    main()

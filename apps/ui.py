"""Streamlit 화면에서 공통으로 사용하는 디자인 요소."""

from __future__ import annotations

import re
from html import escape

import streamlit as st


# 모델은 수식을 \( \) 와 \[ \] 로 감싸 오는데, Streamlit 의 마크다운은 $ 만 수식으로
# 읽는다. 그대로 두면 괄호만 남고 \frac 이나 \mathbb{R} 이 날것으로 보인다.
# 요약본 파일과 딥서치 답변 양쪽에서 같은 일이 생겨 여기에 모아 둔다.
DISPLAY_MATH_PATTERN = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
INLINE_MATH_PATTERN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def _one_line(body: str) -> str:
    """수식 안의 줄바꿈을 없앤다.

    Streamlit 은 $$ 안에 줄바꿈이 있으면 수식으로 읽지 않고, 그러면 마크다운이
    그 줄들을 앞뒤 문단과 한 덩어리로 합쳐 버린다. 수식이 통째로 날것으로 보이고
    소제목까지 문장 사이에 끼는 이유가 이것이다. LaTeX 은 공백을 무시하므로
    한 줄로 눌러도 식의 뜻은 그대로다.
    """
    return " ".join(body.split())


def normalize_math(markdown: str) -> str:
    """LaTeX 식 구분자를 Streamlit 이 읽는 달러 표기로 바꾼다."""
    markdown = DISPLAY_MATH_PATTERN.sub(lambda m: f"$${_one_line(m.group(1))}$$", markdown)
    return INLINE_MATH_PATTERN.sub(lambda m: f"${_one_line(m.group(1))}$", markdown)


def apply_global_styles() -> None:
    """Google Scholar의 간결한 검색 경험을 참고한 공통 스타일을 적용한다."""
    st.markdown(
        """
        <style>
        :root {
            --scholar-blue: #1a73e8;
            --scholar-blue-dark: #1558b0;
            --scholar-green: #2e7d32;
            --scholar-text: #202124;
            --scholar-muted: #5f6368;
            --scholar-border: #dadce0;
            --scholar-surface: #f8f9fa;
        }
        .stApp { background: #fff; color: var(--scholar-text); }
        .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] {
            background: var(--scholar-surface);
            border-right: 1px solid var(--scholar-border);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .42rem .35rem; }
        .sidebar-brand { display:flex; align-items:center; gap:.7rem; padding:.4rem 0 1.35rem; }
        .sidebar-brand-icon {
            display:grid; place-items:center; width:2.3rem; height:2.3rem;
            border-radius:50%; background:#e8f0fe;
        }
        .sidebar-brand strong, .sidebar-brand small { display:block; }
        .sidebar-brand small { color:var(--scholar-muted); font-size:.72rem; margin-top:.1rem; }
        .sidebar-help {
            margin-top:2rem; padding:.9rem 1rem; border-top:1px solid var(--scholar-border);
            color:var(--scholar-muted); font-size:.78rem; line-height:1.7;
        }
        .scholar-hero { text-align:center; padding:3.8rem 1rem 1.8rem; }
        .scholar-logo {
            margin:0; font-size:clamp(2.5rem, 6vw, 4.2rem); font-weight:500;
            letter-spacing:-.08em; line-height:1.08;
        }
        .scholar-logo .blue { color:#4285f4; }
        .scholar-logo .red { color:#ea4335; }
        .scholar-logo .yellow { color:#fbbc05; }
        .scholar-logo .green { color:#34a853; }
        .scholar-logo .dark { color:#5f6368; letter-spacing:-.05em; margin-left:.1em; }
        .scholar-hero p { color:var(--scholar-muted); margin:.75rem 0 0; font-size:.98rem; }
        .page-heading {
            padding:.35rem 0 1.25rem; border-bottom:1px solid var(--scholar-border);
            margin-bottom:1.5rem;
        }
        .page-heading h1 { font-size:1.72rem; font-weight:500; margin:0; color:var(--scholar-text); }
        .page-heading p { color:var(--scholar-muted); margin:.45rem 0 0; }
        .result-card { padding:1rem 0 1.15rem; border-bottom:1px solid #eceff1; }
        .result-title {
            color:var(--scholar-blue); font-size:1.15rem; font-weight:500;
            line-height:1.35; text-decoration:none;
        }
        .result-title:hover { color:var(--scholar-blue-dark); text-decoration:underline; }
        .result-meta { color:var(--scholar-green); font-size:.85rem; margin:.35rem 0; }
        .result-abstract { color:#3c4043; font-size:.9rem; line-height:1.55; margin:0; }
        .paper-detail-card {
            border:1px solid var(--scholar-border); border-radius:.65rem; padding:1.35rem 1.5rem;
            margin:1rem 0; background:#fff; box-shadow:0 1px 2px rgba(60,64,67,.08);
        }
        .paper-detail-card h2 {
            color:var(--scholar-blue); font-size:1.35rem; font-weight:500; margin:0 0 .55rem;
        }
        .paper-detail-card p { color:var(--scholar-muted); margin:.2rem 0; }
        .empty-guide {
            padding:2.5rem 1rem; text-align:center; color:var(--scholar-muted);
            background:var(--scholar-surface); border-radius:.65rem;
        }
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            border-color:var(--scholar-border);
        }
        div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {
            border-color:var(--scholar-blue); box-shadow:0 0 0 1px var(--scholar-blue);
        }
        [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
            background:var(--scholar-blue) !important; border-color:var(--scholar-blue) !important;
            color:#fff !important;
        }
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-primaryFormSubmit"]:hover {
            background:var(--scholar-blue-dark) !important;
            border-color:var(--scholar-blue-dark) !important;
        }
        .stButton > button, .stFormSubmitButton > button { border-radius:.3rem; font-weight:500; }
        [data-testid="stSlider"] [role="slider"] { background:var(--scholar-blue) !important; }
        [data-testid="stChatMessage"] {
            border:1px solid #eceff1; border-radius:.65rem; background:#fff; margin-bottom:.65rem;
        }
        @media (max-width:700px) {
            .scholar-hero { padding-top:1.7rem; }
            .block-container { padding-top:1rem; }
            .paper-detail-card { padding:1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_heading(title: str, description: str) -> None:
    """모든 화면에서 동일한 형식의 제목과 안내를 표시한다."""
    st.markdown(
        f"""
        <div class="page-heading">
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

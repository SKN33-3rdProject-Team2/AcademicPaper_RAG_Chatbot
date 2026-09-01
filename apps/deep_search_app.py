"""LangGraph Supervisor와 연결된 Streamlit 채팅 화면."""

from __future__ import annotations

import json
from uuid import uuid4

import streamlit as st

from apps.ui import normalize_math, render_page_heading
from feature.search_list import LocalLibraryBot
from orchestration.graph import build_graph
from orchestration.state import initial_state
from tools import PROJECT_DIR


MESSAGES_KEY = "supervisor_chat_messages"
SESSION_KEY = "supervisor_chat_session"
TURN_KEY = "supervisor_chat_turn"
PAPER_IDS_KEY = "supervisor_paper_ids"
SELECTED_PAPER_KEY = "supervisor_selected_paper_id"
EXTRACTED_PAPERS_PATH = PROJECT_DIR / "data" / "paper_extract" / "extracted_papers.json"


@st.cache_resource(show_spinner=False)
def _get_graph():
    """Graph 구조는 재사용하고 각 질문은 독립된 checkpoint로 실행한다."""
    return build_graph()


def _reset_chat() -> None:
    st.session_state[SESSION_KEY] = uuid4().hex[:10]
    st.session_state[TURN_KEY] = 0
    st.session_state[PAPER_IDS_KEY] = []
    st.session_state[SELECTED_PAPER_KEY] = None
    st.session_state[MESSAGES_KEY] = [
        {
            "role": "assistant",
            "content": (
                "논문 검색, 서재 조회, 번역·요약, RAG 및 심층 분석을 도와드립니다. "
                "원하는 작업을 자연어로 말씀해 주세요."
            ),
        }
    ]


def _run_supervisor(prompt: str) -> tuple[str, list[dict]]:
    turn = st.session_state[TURN_KEY]
    thread_id = f"streamlit-{st.session_state[SESSION_KEY]}-{turn}"
    st.session_state[TURN_KEY] = turn + 1
    selected_paper_id = st.session_state.get(SELECTED_PAPER_KEY)
    paper_ids = [selected_paper_id] if selected_paper_id else list(st.session_state[PAPER_IDS_KEY])
    normalized = prompt.casefold()
    list_signals = ("보유", "서재", "저장된", "목록", "리스트", "library")
    action_signals = ("다운로드", "번역", "요약", "검색해", "찾아줘")
    # 현재 Supervisor의 library 노드는 ID가 없으면 제목 키워드 검색을 한다.
    # 단순 전체 목록 요청일 때만 기존 LocalLibraryBot의 전체 ID를 전달한다.
    if (
        not paper_ids
        and any(signal in normalized for signal in list_signals)
        and not any(signal in normalized for signal in action_signals)
    ):
        paper_ids = LocalLibraryBot().get_all_json_ids()

    state = initial_state(
        prompt,
        thread_id=thread_id,
        paper_ids=paper_ids,
    )
    result = _get_graph().invoke(
        state,
        config={
            "configurable": {"thread_id": thread_id},
            "run_name": "streamlit-academic-paper-stategraph",
            "tags": ["streamlit", "academic-paper"],
        },
    )
    if result.get("paper_ids"):
        st.session_state[PAPER_IDS_KEY] = list(result["paper_ids"])
    return str(result.get("response") or "답변을 생성하지 못했습니다."), list(
        result.get("sources") or []
    )


def _extracted_paper_options() -> dict[str, str]:
    """본문 기반 Q&A 대상으로 선택할 수 있는 추출 논문 목록을 읽는다."""
    if not EXTRACTED_PAPERS_PATH.exists():
        return {}
    try:
        data = json.loads(EXTRACTED_PAPERS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(paper_id): str(paper.get("title") or paper_id)
        for paper_id, paper in data.items()
        if isinstance(paper, dict)
    }


def render_deep_search_page() -> None:
    """대화 내역을 표시하고 질문을 Supervisor Graph로 전달한다."""
    render_page_heading(
        "딥서치",
        "Supervisor가 요청을 분석하여 필요한 논문 기능을 순서대로 실행합니다.",
    )
    if SESSION_KEY not in st.session_state:
        _reset_chat()

    paper_options = _extracted_paper_options()
    selected_paper_id = st.selectbox(
        "질문할 논문 선택",
        options=[None, *paper_options],
        format_func=lambda paper_id: (
            "선택하지 않음" if paper_id is None else paper_options[paper_id]
        ),
        help="본문 기반 질의응답은 한 번에 논문 한 편만 대상으로 합니다.",
    )
    st.session_state[SELECTED_PAPER_KEY] = selected_paper_id

    controls, guide = st.columns([1.7, 8.3])
    with controls:
        new_chat = st.button("새 대화 시작", use_container_width=True)
    with guide:
        st.caption("대화 기록과 선택된 논문 정보는 현재 브라우저 세션에서 유지됩니다.")
    if new_chat:
        _reset_chat()
        st.rerun()

    st.markdown("**이렇게 요청해 보세요**")
    examples = [
        "보유 중인 논문 목록을 보여줘",
        "Attention 관련 논문을 찾아줘",
        "저장된 논문의 핵심 내용을 설명해줘",
    ]
    example_columns = st.columns(3)
    selected_example = None
    for column, example in zip(example_columns, examples):
        with column:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                selected_example = example

    messages = st.session_state[MESSAGES_KEY]
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                st.caption(
                    f"출처: {source.get('title') or source.get('paper_id') or source.get('label')}"
                )

    prompt = selected_example or st.chat_input("논문에 대해 질문해 주세요.")
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Supervisor가 필요한 작업을 실행하고 있습니다..."):
                response, sources = _run_supervisor(prompt)
        except Exception as exc:
            response, sources = f"요청 처리 중 오류가 발생했습니다: {exc}", []
        # 답변은 모델이 즉석에서 만들어 \( \) 표기가 섞여 온다. 화면에 뿌리기 전에
        # 맞춰 두면 대화 기록에도 고쳐진 채로 남아 다시 그릴 때 또 깨지지 않는다.
        response = normalize_math(response)
        st.markdown(response)
        for source in sources:
            st.caption(
                f"출처: {source.get('title') or source.get('paper_id') or source.get('label')}"
            )

    messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )

"""LangGraph Supervisor와 연결된 Streamlit 채팅 화면."""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

from apps.ui import render_page_heading
from feature.search_list import LocalLibraryBot
from orchestration.graph import build_graph
from orchestration.state import initial_state


MESSAGES_KEY = "supervisor_chat_messages"
SESSION_KEY = "supervisor_chat_session"
TURN_KEY = "supervisor_chat_turn"
PAPER_IDS_KEY = "supervisor_paper_ids"


@st.cache_resource(show_spinner=False)
def _get_graph():
    """Graph 구조는 재사용하고 각 질문은 독립된 checkpoint로 실행한다."""
    return build_graph()


def _reset_chat() -> None:
    st.session_state[SESSION_KEY] = uuid4().hex[:10]
    st.session_state[TURN_KEY] = 0
    st.session_state[PAPER_IDS_KEY] = []
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
    paper_ids = list(st.session_state[PAPER_IDS_KEY])
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


def render_deep_search_page() -> None:
    """대화 내역을 표시하고 질문을 Supervisor Graph로 전달한다."""
    render_page_heading(
        "딥서치",
        "Supervisor가 요청을 분석하여 필요한 논문 기능을 순서대로 실행합니다.",
    )
    if SESSION_KEY not in st.session_state:
        _reset_chat()

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
        st.markdown(response)
        for source in sources:
            st.caption(
                f"출처: {source.get('title') or source.get('paper_id') or source.get('label')}"
            )

    messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )

"""Supervisor planning and routing for the paper workflow."""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from orchestration.state import Route, WorkflowState


ExecutableRoute = Literal[
    "keyword",
    "search",
    "library",
    "download",
    "extract",
    "translate",
    "summarize",
    "rag",
    "deep_research",
]


class SupervisorDecision(BaseModel):
    """A validated execution plan emitted by the supervisor."""

    steps: list[ExecutableRoute] = Field(min_length=1, max_length=8)
    reason: str


SUPERVISOR_PROMPT = """You plan work for an academic-paper assistant.
Return the shortest valid ordered list of node names.

Nodes:
- keyword: generate arXiv keywords from a research topic
- search: search arXiv; normally place keyword immediately before it
- library: list or search papers already saved locally
- download: download already selected/search-result papers
- extract: extract PDF text; required before translate
- translate: translate extracted Markdown; required before summarize
- summarize: summarize translated Markdown and store it in ChromaDB
- rag: answer a question using only summaries already stored in ChromaDB
- deep_research: deeply analyze the paper selected by RAG; use only after rag

Rules:
1. For a new external search use [keyword, search].
2. For translation use [extract, translate] unless extraction data exists.
3. For summary use [extract, translate, summarize] unless earlier artifacts exist.
4. RAG is QA only; never use RAG as a translation step.
5. Do not invent a download step if no selected/search-result papers exist.
6. Prefer library for list/search requests about locally saved papers.
7. For deep analysis or comparison, start with [rag]. The supervisor will send
   a retrieved paper to deep_research only after RAG finds a source.
8. A request that chains multiple stages (e.g. "find the latest 5 LLM papers,
   translate and summarize them, then explain them") is ONE plan, not
   separate requests — emit the full ordered chain in one call, for example
   [keyword, search, download, extract, translate, summarize, rag]. Only
   include the stages actually implied by the request and skip stages whose
   artifacts already exist per "Available state".
9. Treat conversational questions about which papers the assistant can explain
   as requests to inspect existing RAG content first. Never start an external
   search or download for those questions unless RAG returns no source.
"""


class SupervisorRouter:
    """Hybrid router: structured LLM planning with a safe rule fallback."""

    def __init__(self, llm: Any | None = None, *, use_llm: bool | None = None) -> None:
        self._llm = llm
        self._use_llm = (
            os.getenv("SUPERVISOR_USE_LLM", "true").casefold() == "true"
            if use_llm is None
            else use_llm
        )

    @property
    def llm(self):
        if self._llm is None:
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            self._llm = ChatOpenAI(model=model, temperature=0)
        return self._llm

    @staticmethod
    def _rule_decision(state: WorkflowState) -> SupervisorDecision | None:
        query = state["query"].casefold()
        has_extraction = bool(state.get("extracted_records"))
        has_translation = bool(state.get("translated_paths"))
        has_candidates = bool(
            state.get("selected_papers")
            or state.get("search_results")
            or state.get("library_results")
        )

        # In a chatbot, "설명 가능한 논문이 뭐가 있어?" asks what the
        # assistant can currently explain. It is not an instruction to search,
        # download, and process ten new papers. Inspect indexed RAG content
        # first; the graph will hand a retrieved source to Deep Research and
        # will run the external rebuild pipeline only when RAG has no source.
        asks_explainable_inventory = (
            "논문" in query
            and any(
                phrase in query
                for phrase in ("설명 가능한", "설명할 수 있는", "설명해줄 수 있는")
            )
            and any(term in query for term in ("뭐가", "무엇", "어떤", "있어", "있나", "보여"))
        )
        if asks_explainable_inventory:
            return SupervisorDecision(
                steps=["rag"],
                reason="보유 문서 중 설명 가능한 논문 확인",
            )

        # A request can chain multiple stages in one sentence (e.g. "찾아서
        # 번역 요약해주고 설명해줘" = search + translate + summarize + explain).
        # Detect that BEFORE the single-purpose keyword checks below, which
        # would otherwise stop at whichever keyword happens to match first
        # and silently drop the rest of the request.
        wants_new_papers = any(
            term in query
            for term in ("arxiv", "외부 검색", "논문 찾아", "찾아서", "찾아줘", "검색해", "다운로드", "download")
        )
        wants_translate = any(term in query for term in ("번역", "translate"))
        wants_summarize = any(term in query for term in ("요약", "summar", "summary"))
        wants_qa = any(term in query for term in ("근거", "출처", "질문", "설명해"))
        wants_deep = any(
            term in query for term in ("딥리서치", "심층", "비교", "분석", "deep research")
        )
        if wants_new_papers and (wants_translate or wants_summarize or wants_qa or wants_deep):
            steps: list[ExecutableRoute] = []
            if any(
                term in query
                for term in ("arxiv", "외부 검색", "논문 찾아", "찾아서", "찾아줘", "검색해")
            ):
                steps += ["keyword", "search"]
            if wants_translate or wants_summarize:
                if not has_extraction:
                    steps.append("download")
                    steps.append("extract")
                steps.append("translate")
                if wants_summarize:
                    steps.append("summarize")
            elif any(term in query for term in ("다운로드", "download")):
                steps.append("download")
            if wants_qa or wants_deep:
                steps.append("rag")
            ordered: list[ExecutableRoute] = []
            for step in steps:
                if step not in ordered:
                    ordered.append(step)
            return SupervisorDecision(steps=ordered[:8], reason="검색부터 설명까지 이어지는 복합 요청")

        explicit_qa_signals = ("근거", "출처", "질문", "설명해")
        stored_content_signals = ("저장", "요약", "서재")
        if any(term in query for term in explicit_qa_signals) or (
            "rag" in query and any(term in query for term in stored_content_signals)
        ):
            return SupervisorDecision(steps=["rag"], reason="저장된 요약 기반 질의응답")
        if any(term in query for term in ("번역", "translate")):
            steps = [] if has_extraction else ["extract"]
            steps.append("translate")
            return SupervisorDecision(steps=steps, reason="번역 요청")
        if any(term in query for term in ("요약", "summar", "summary")):
            steps = []
            if not has_translation:
                if not has_extraction:
                    steps.append("extract")
                steps.append("translate")
            steps.append("summarize")
            return SupervisorDecision(steps=steps, reason="요약 파이프라인 요청")
        if any(term in query for term in ("arxiv", "외부 검색", "논문 찾아", "찾아서", "찾아줘", "검색해")):
            return SupervisorDecision(steps=["keyword", "search"], reason="외부 논문 검색 요청")
        if any(term in query for term in ("다운로드", "download")):
            steps = ["download"] if has_candidates else ["library", "download"]
            return SupervisorDecision(steps=steps, reason="논문 다운로드 요청")
        if any(term in query for term in ("딥리서치", "심층", "비교", "분석", "deep research")):
            return SupervisorDecision(steps=["rag"], reason="저장 문서 검색 후 심층 분석")
        if any(term in query for term in ("서재", "저장된", "목록", "리스트", "library")):
            return SupervisorDecision(steps=["library"], reason="로컬 서재 요청")
        if any(term in query for term in ("rag", "근거", "출처", "질문", "설명해")):
            return SupervisorDecision(steps=["rag"], reason="저장된 요약 기반 질의응답")
        return None

    @classmethod
    def _fallback(cls, state: WorkflowState) -> SupervisorDecision:
        return cls._rule_decision(state) or SupervisorDecision(
            steps=["rag"],
            reason="저장 문서 검색 후 질의응답",
        )

    def decide(self, state: WorkflowState) -> SupervisorDecision:
        rule_decision = self._rule_decision(state)
        if rule_decision is not None:
            return rule_decision
        if not self._use_llm:
            return self._fallback(state)
        inventory = {
            "paper_ids": state.get("paper_ids", []),
            "has_search_results": bool(state.get("search_results")),
            "has_selected_papers": bool(state.get("selected_papers")),
            "has_extracted_records": bool(state.get("extracted_records")),
            "has_translated_paths": bool(state.get("translated_paths")),
        }
        try:
            structured = self.llm.with_structured_output(SupervisorDecision)
            return structured.invoke(
                f"{SUPERVISOR_PROMPT}\n\nUser request: {state['query']}\nAvailable state: {inventory}"
            )
        except Exception:
            return self._fallback(state)


def next_route(state: WorkflowState) -> Route:
    return state.get("route", "finish")

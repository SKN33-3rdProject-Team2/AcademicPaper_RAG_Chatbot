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

    steps: list[ExecutableRoute] = Field(min_length=1, max_length=6)
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

        explicit_qa_signals = ("근거", "출처", "질문", "설명해")
        stored_content_signals = ("저장", "요약", "서재")
        if any(term in query for term in explicit_qa_signals) or (
            "rag" in query and any(term in query for term in stored_content_signals)
        ):
            return SupervisorDecision(steps=["rag"], reason="저장된 요약 기반 질의응답")
        if any(term in query for term in ("번역", "translate")):
            steps: list[ExecutableRoute] = [] if has_extraction else ["extract"]
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
        if any(term in query for term in ("arxiv", "외부 검색", "논문 찾아", "검색해")):
            return SupervisorDecision(steps=["keyword", "search"], reason="외부 논문 검색 요청")
        if any(term in query for term in ("다운로드", "download")):
            steps = ["download"] if state.get("selected_papers") else ["library", "download"]
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

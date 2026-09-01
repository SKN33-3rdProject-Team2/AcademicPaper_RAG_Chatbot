"""RAG chain that answers from extracted paper-text chunks."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from orchestration.state import WorkflowState
from services import PROJECT_ROOT
from services.fulltext_vector_store import FullTextStoreError


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 추출된 논문 본문 근거만 사용하는 RAG 답변자입니다. "
            "제공된 근거 밖의 사실을 만들지 마세요. 각 핵심 주장 뒤에 [S1] 형식으로 "
            "출처 번호를 표시하세요. 근거가 부족하면 부족하다고 명확히 답하세요. "
            "수식은 반드시 달러 기호로 감쌉니다. 문장 안에서는 $...$ 로, "
            "독립된 줄에서는 $$...$$ 로 쓰고, \\( \\) 나 \\[ \\] 표기는 쓰지 않습니다.",
        ),
        ("human", "질문:\n{question}\n\n검색된 근거:\n{context}"),
    ]
)


def _default_store():
    from services.fulltext_vector_store import ChromaFullTextStore

    return ChromaFullTextStore()


def _default_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


class SummaryRAGChain:
    """LCEL retriever -> prompt -> LLM -> answer + source payload."""

    def __init__(
        self,
        *,
        store_factory: Callable[[], Any] = _default_store,
        llm: Any | None = None,
        top_k: int = 5,
    ) -> None:
        self._store_factory = store_factory
        self._store: Any | None = None
        self._llm = llm or _default_llm()
        self._top_k = top_k
        answer_chain = RAG_PROMPT | self._llm | StrOutputParser()
        with_answer = RunnablePassthrough.assign(answer=answer_chain)
        no_evidence = RunnableLambda(
            lambda payload: {
                **payload,
                "answer": "저장된 논문 본문에서 질문에 답할 근거를 찾지 못했습니다.",
            }
        )
        self.chain = RunnableLambda(self._retrieve) | RunnableBranch(
            (lambda payload: bool(payload["sources"]), with_answer),
            no_evidence,
        )

    @property
    def store(self):
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    def _retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload["question"])
        paper_id = payload.get("paper_id")
        results = self.store.search(question, limit=self._top_k, paper_id=paper_id)
        sources: list[dict[str, Any]] = []
        context_parts: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = dict(result.get("metadata") or {})
            document = str(result.get("document") or "")
            source = {
                "label": f"S{index}",
                "id": str(result.get("id") or ""),
                "paper_id": str(metadata.get("paper_id") or ""),
                "title": str(metadata.get("title") or ""),
                "section": str(metadata.get("section") or ""),
                "distance": result.get("distance"),
                "excerpt": document[:500],
            }
            sources.append(source)
            context_parts.append(
                f"[S{index}] title={source['title']} section={source['section']}\n{document}"
            )
        return {
            "question": question,
            "context": "\n\n".join(context_parts),
            "sources": sources,
        }

    def invoke(self, question: str, *, paper_id: str | None = None) -> dict[str, Any]:
        try:
            result = self.chain.invoke({"question": question, "paper_id": paper_id})
        except FullTextStoreError:
            # 본문 DB·색인 모델이 준비되지 않았을 때도 환각 답변 대신 안내한다.
            result = {"sources": [], "answer": ""}
        if not result["sources"]:
            reference_hint = self._reference_hint(paper_id)
            return {
                "answer": (
                    "해당 논문의 본문 데이터가 없어 질문에 답할 수 없습니다. "
                    f"{reference_hint}"
                ),
                "sources": [],
            }
        return {
            "answer": self._append_original_evidence(result["answer"], result["sources"]),
            "sources": result["sources"],
        }

    @staticmethod
    def _append_original_evidence(answer: str, sources: list[dict[str, Any]]) -> str:
        """답변에 사용자가 검증할 수 있는 원문 발췌를 함께 표시한다."""
        evidence_lines = ["\n\n### 원문 근거\n"]
        for source in sources:
            label = str(source.get("label") or "S?")
            title = str(source.get("title") or "제목 없음")
            section = str(source.get("section") or "본문")
            excerpt = str(source.get("excerpt") or "").strip().replace("\n", " ")
            evidence_lines.extend(
                [
                    f"- [{label}] {title} · {section}",
                    f"  > {excerpt}",
                ]
            )
        return str(answer).rstrip() + "\n".join(evidence_lines)

    @staticmethod
    def _reference_hint(paper_id: str | None) -> str:
        if not paper_id:
            return "답변하려는 논문을 선택한 뒤 참고문헌을 확인해 주세요."
        ref_db = PROJECT_ROOT / "data" / "paper_extract" / "extracted_papers_ref.db"
        if not ref_db.exists():
            return "해당 논문의 참고문헌을 원문에서 확인해 주세요."
        try:
            with sqlite3.connect(ref_db) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM extracted_ref WHERE paper_id = ?", (paper_id,)
                ).fetchone()[0]
        except sqlite3.Error:
            count = 0
        if count:
            return f"저장된 참고문헌 {count}건을 참조해 주세요."
        return "해당 논문의 참고문헌을 원문에서 확인해 주세요."


class RAGNode:
    def __init__(
        self,
        chain: SummaryRAGChain | None = None,
        *,
        chain_factory: Callable[[], SummaryRAGChain] = SummaryRAGChain,
    ) -> None:
        self._chain = chain
        self._chain_factory = chain_factory

    @property
    def chain(self) -> SummaryRAGChain:
        if self._chain is None:
            self._chain = self._chain_factory()
        return self._chain

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        paper_ids = state.get("paper_ids", [])
        if len(paper_ids) != 1:
            return {
                "rag_answer": "질문할 논문을 한 편만 선택해 주세요.",
                "response": "질문할 논문을 한 편만 선택해 주세요.",
                "sources": [],
                "rag_selection_required": True,
                "node_history": ["rag"],
            }
        result = self.chain.invoke(
            state["query"],
            paper_id=paper_ids[0],
        )
        return {
            "rag_answer": result["answer"],
            "response": result["answer"],
            "sources": result["sources"],
            "rag_selection_required": False,
            "node_history": ["rag"],
        }

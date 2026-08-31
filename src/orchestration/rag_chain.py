"""RAG-only LangChain chain over the existing summary vector store."""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from orchestration.state import WorkflowState


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 논문 요약 근거만 사용하는 RAG 답변자입니다. "
            "제공된 근거 밖의 사실을 만들지 마세요. 각 핵심 주장 뒤에 [S1] 형식으로 "
            "출처 번호를 표시하세요. 근거가 부족하면 부족하다고 명확히 답하세요.",
        ),
        ("human", "질문:\n{question}\n\n검색된 근거:\n{context}"),
    ]
)


def _default_store():
    from services.summary_vector_store import ChromaSummaryStore

    return ChromaSummaryStore()


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
                "answer": "저장된 논문 요약에서 질문에 답할 근거를 찾지 못했습니다.",
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
        result = self.chain.invoke({"question": question, "paper_id": paper_id})
        return {"answer": result["answer"], "sources": result["sources"]}


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
        result = self.chain.invoke(
            state["query"],
            paper_id=paper_ids[0] if len(paper_ids) == 1 else None,
        )
        return {
            "rag_answer": result["answer"],
            "response": result["answer"],
            "sources": result["sources"],
            "node_history": ["rag"],
        }

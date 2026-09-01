"""Adapters that expose existing chatbot classes as StateGraph nodes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from orchestration.state import WorkflowState


class NodeExecutionError(RuntimeError):
    """Raised when an existing team component cannot satisfy the node contract."""


class StateNode(Protocol):
    def __call__(self, state: WorkflowState) -> dict[str, Any]: ...


def _record(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise NodeExecutionError(f"지원하지 않는 결과 형식입니다: {type(value).__name__}")


def _default_keyword_tool():
    from tools.keyword_tool import KeywordTool

    return KeywordTool()


def _default_search_bot():
    from feature.search import ArxivSearchBot

    return ArxivSearchBot()


def _default_library_bot():
    from feature.search_list import LocalLibraryBot

    return LocalLibraryBot()


def _default_extractor():
    from feature.paper_extractor import PaperExtractor

    return PaperExtractor()


def _default_translator():
    from tools.translation_tool import TranslateTool

    return TranslateTool()


def _default_summarizer():
    from tools.summary_tool import SummaryTool

    return SummaryTool()


def _default_deep_research_agent():
    from feature.deep_research import DeepResearchBot

    # with_openai 가 저장소·검색기·답변기를 한꺼번에 엮어 준다. 생성자를 직접 부르면
    # repository 를 인자로 받아야 해서, 기본값으로 쓰기에는 이쪽이 맞다.
    return DeepResearchBot.with_openai()


class KeywordNode:
    def __init__(self, factory: Callable[[], Any] = _default_keyword_tool) -> None:
        self._factory = factory
        self._tool: Any | None = None

    @property
    def tool(self):
        if self._tool is None:
            self._tool = self._factory()
        return self._tool

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        topic = state["query"]
        if int(state.get("retry_counts", {}).get("search", 0)) > 0:
            previous = ", ".join(state.get("keywords", []))
            topic = (
                f"{topic}\n"
                "이전 검색 결과가 없었습니다. 같은 의미를 유지하되 "
                f"다음 표현과 겹치지 않는 대체 학술 용어를 생성하세요: {previous}"
            )
        result = self.tool.generate_keywords(topic)
        keywords = [str(item) for item in result.get("keywords", []) if str(item).strip()]
        if not keywords:
            raise NodeExecutionError("검색 키워드를 생성하지 못했습니다.")
        return {"keywords": keywords, "node_history": ["keyword"]}


class ArxivSearchNode:
    def __init__(
        self,
        factory: Callable[[], Any] = _default_search_bot,
        *,
        max_results: int = 10,
    ) -> None:
        self._factory = factory
        self._bot: Any | None = None
        self._max_results = max_results

    @property
    def bot(self):
        if self._bot is None:
            self._bot = self._factory()
        return self._bot

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        terms = state.get("keywords") or [state["query"]]
        query = " OR ".join(f'"{term}"' for term in terms)
        papers = self.bot.search_papers(query, max_results=self._max_results)
        return {"search_results": list(papers), "node_history": ["search"]}


class LocalLibraryNode:
    def __init__(self, factory: Callable[[], Any] = _default_library_bot) -> None:
        self._factory = factory
        self._bot: Any | None = None

    @property
    def bot(self):
        if self._bot is None:
            self._bot = self._factory()
        return self._bot

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        paper_ids = list(state.get("paper_ids", []))
        if not paper_ids:
            paper_ids = self.bot.search_json(state["query"])
        papers = self.bot.fetch_full_data_from_db(paper_ids) if paper_ids else []
        return {
            "paper_ids": [str(paper["id"]) for paper in papers],
            "library_results": papers,
            "node_history": ["library"],
        }


class DownloadNode:
    def __init__(self, factory: Callable[[], Any] = _default_library_bot) -> None:
        self._factory = factory
        self._bot: Any | None = None

    @property
    def bot(self):
        if self._bot is None:
            self._bot = self._factory()
        return self._bot

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        papers = (
            state.get("selected_papers")
            or state.get("library_results")
            or state.get("search_results")
            or []
        )
        if not papers:
            raise NodeExecutionError("다운로드할 논문이 선택되지 않았습니다.")
        paths: list[str] = []
        paper_ids: list[str] = []
        for paper in papers:
            path = self.bot.download_pdf(paper)
            if path:
                paths.append(str(path))
                if paper.get("id"):
                    paper_ids.append(str(paper["id"]))
        return {
            "paper_ids": paper_ids or list(state.get("paper_ids", [])),
            "downloaded_paths": paths,
            "node_history": ["download"],
        }


class ExtractNode:
    def __init__(self, factory: Callable[[], Any] = _default_extractor) -> None:
        self._factory = factory
        self._extractor: Any | None = None

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = self._factory()
        return self._extractor

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        paper_ids = list(state.get("paper_ids", []))
        if not paper_ids:
            raise NodeExecutionError("추출할 paper_id가 없습니다.")
        results = self.extractor.extract_many(paper_ids)
        records = [_record(result) for result in results]
        if not records:
            raise NodeExecutionError("논문 본문을 추출하지 못했습니다.")
        return {"extracted_records": records, "node_history": ["extract"]}


class TranslateNode:
    def __init__(
        self,
        translator_factory: Callable[[], Any] = _default_translator,
        extractor_factory: Callable[[], Any] = _default_extractor,
    ) -> None:
        self._translator_factory = translator_factory
        self._extractor_factory = extractor_factory
        self._translator: Any | None = None
        self._extractor: Any | None = None

    @property
    def translator(self):
        if self._translator is None:
            self._translator = self._translator_factory()
        return self._translator

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = self._extractor_factory()
        return self._extractor

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        records = list(state.get("extracted_records", []))
        if not records:
            records = [
                record
                for paper_id in state.get("paper_ids", [])
                if (record := self.extractor.get(paper_id))
            ]
        if not records:
            raise NodeExecutionError("번역 전에 논문 본문 추출이 필요합니다.")
        paths = [
            str(
                self.translator.translate_paper(
                    record["content"],
                    paper_id=str(record.get("id", "")),
                    title=str(record.get("title", "")),
                )
            )
            for record in records
        ]
        return {"translated_paths": paths, "node_history": ["translate"]}


class SummaryNode:
    def __init__(self, factory: Callable[[], Any] = _default_summarizer) -> None:
        self._factory = factory
        self._tool: Any | None = None

    @property
    def tool(self):
        if self._tool is None:
            self._tool = self._factory()
        return self._tool

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        paths = [Path(path) for path in state.get("translated_paths", [])]
        if not paths:
            raise NodeExecutionError("요약 전에 번역 Markdown 생성이 필요합니다.")
        summaries = [_record(self.tool.summarize_file(path)) for path in paths]
        for summary in summaries:
            if summary.get("markdown_path") is not None:
                summary["markdown_path"] = str(summary["markdown_path"])
        return {"summaries": summaries, "node_history": ["summarize"]}


class DeepResearchNode:
    def __init__(self, factory: Callable[[], Any] = _default_deep_research_agent) -> None:
        self._factory = factory
        self._agent: Any | None = None

    @property
    def agent(self):
        if self._agent is None:
            self._agent = self._factory()
        return self._agent

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        rag_sources = [
            source
            for source in state.get("sources", [])
            if isinstance(source, dict)
        ]
        if not rag_sources:
            raise NodeExecutionError("Deep Research에 전달할 RAG 문서가 없습니다.")

        selected: dict[str, Any] | None = None
        selected_source: dict[str, Any] | None = None
        for source in rag_sources:
            candidates = (source.get("paper_id"), source.get("title"))
            for candidate in candidates:
                selection = str(candidate or "").strip()
                if not selection:
                    continue
                result = self.agent.select_paper(selection)
                if result.get("status") == "selected":
                    selected = result
                    selected_source = source
                    break
            if selected is not None:
                break

        if selected is None:
            result = {
                "status": "selection_required",
                "message": "RAG가 찾은 문서를 Deep Research 대상 논문으로 선택하지 못했습니다.",
                "sources": [],
            }
        else:
            result = self.agent.ask(state["query"])

        response = str(result.get("answer") or result.get("message") or "")
        deep_sources = list(result.get("sources") or [])
        paper = result.get("paper") or selected.get("paper") if selected else {}
        paper_id = str(
            (paper or {}).get("id")
            or (selected_source or {}).get("paper_id")
            or ""
        )

        payload: dict[str, Any] = {
            "response": response,
            "deep_research_status": str(result.get("status") or "unknown"),
            "deep_research_answer": response,
            "deep_research_sources": deep_sources,
            "deep_research_paper_id": paper_id,
            "node_history": ["deep_research"],
        }
        return payload

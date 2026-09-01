"""StateGraph wiring for all paper chatbot classes and the RAG chain."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from orchestration.adapters import (
    ArxivSearchNode,
    DeepResearchNode,
    DownloadNode,
    ExtractNode,
    KeywordNode,
    LocalLibraryNode,
    SummaryNode,
    TranslateNode,
)
from orchestration.rag_chain import RAGNode
from orchestration.routing import SupervisorRouter, next_route
from orchestration.state import Route, WorkflowState


def _guarded(name: str, node: Callable[[WorkflowState], dict[str, Any]]):
    @traceable(name=f"node.{name}", run_type="tool")
    def invoke(state: WorkflowState) -> dict[str, Any]:
        try:
            return node(state)
        except Exception as exc:
            return {
                "errors": [f"{name}: {type(exc).__name__}: {exc}"],
                "remaining_steps": [],
                "node_history": [f"{name}:failed"],
            }

    return invoke


def _format_response(state: WorkflowState) -> str:
    if state.get("errors"):
        return "작업을 완료하지 못했습니다. " + " | ".join(state["errors"])
    if state.get("response"):
        return state["response"]

    # search_results/library_results/etc. are conversational memory that stays
    # in state across turns, so only surface the artifact whose node actually
    # ran THIS turn — otherwise a later turn re-prints an earlier turn's result.
    history = state.get("node_history", [])
    if "summarize" in history and state.get("summaries"):
        paths = [item.get("markdown_path") for item in state["summaries"] if item.get("markdown_path")]
        return f"논문 {len(state['summaries'])}편의 요약을 완료했습니다." + (
            "\n" + "\n".join(str(path) for path in paths) if paths else ""
        )
    if "translate" in history and state.get("translated_paths"):
        return "번역을 완료했습니다.\n" + "\n".join(state["translated_paths"])
    if "extract" in history and state.get("extracted_records"):
        return f"논문 {len(state['extracted_records'])}편의 본문 추출을 완료했습니다."
    if "download" in history and state.get("downloaded_paths"):
        return "다운로드를 완료했습니다.\n" + "\n".join(state["downloaded_paths"])
    if "search" in history and state.get("search_results"):
        papers = state["search_results"]
    elif "library" in history and state.get("library_results"):
        papers = state["library_results"]
    else:
        papers = []
    if papers:
        lines = [f"{index}. {paper.get('title', '제목 없음')}" for index, paper in enumerate(papers, 1)]
        return "\n".join(lines)
    return "요청을 처리했지만 반환할 결과가 없습니다."


def build_graph(
    *,
    router: SupervisorRouter | None = None,
    nodes: dict[str, Callable[[WorkflowState], dict[str, Any]]] | None = None,
    checkpointer: Any | None = None,
    interrupt_before: Iterable[str] = (),
):
    """Compile the workflow. Injected nodes keep tests offline and deterministic."""

    supervisor_router = router or SupervisorRouter()
    graph_nodes = nodes or {
        "keyword": KeywordNode(),
        "search": ArxivSearchNode(),
        "library": LocalLibraryNode(),
        "download": DownloadNode(),
        "extract": ExtractNode(),
        "translate": TranslateNode(),
        "summarize": SummaryNode(),
        "rag": RAGNode(),
        "deep_research": DeepResearchNode(),
    }

    def dispatch(
        state: WorkflowState,
        route: Route,
        remaining: list[Route],
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Validate prerequisites and rewrite the remaining plan when needed."""

        history = list(state.get("node_history", []))
        last_node = history[-1] if history else ""

        if route == "translate" and not state.get("extracted_records"):
            if last_node == "extract":
                return {
                    "route": "finish",
                    "route_reason": "본문 추출 결과가 없어 번역을 중단",
                    "remaining_steps": [],
                    "errors": ["번역에 필요한 본문 추출 결과가 없습니다."],
                }
            return {
                "route": "extract",
                "route_reason": "번역 입력이 없어 본문 추출을 먼저 실행",
                "remaining_steps": ["translate", *remaining],
            }

        if route == "summarize" and not state.get("translated_paths"):
            if state.get("extracted_records"):
                return {
                    "route": "translate",
                    "route_reason": "요약 입력이 없어 번역을 먼저 실행",
                    "remaining_steps": ["summarize", *remaining],
                }
            if last_node == "extract":
                return {
                    "route": "finish",
                    "route_reason": "본문 추출 결과가 없어 요약을 중단",
                    "remaining_steps": [],
                    "errors": ["요약에 필요한 본문 추출 결과가 없습니다."],
                }
            return {
                "route": "extract",
                "route_reason": "요약 입력이 없어 추출·번역 단계를 추가",
                "remaining_steps": ["translate", "summarize", *remaining],
            }

        return {
            "route": route,
            "route_reason": reason,
            "remaining_steps": remaining,
        }

    @traceable(name="supervisor.plan", run_type="chain")
    def supervisor(state: WorkflowState) -> dict[str, Any]:
        if state.get("errors"):
            return {"route": "finish", "route_reason": "노드 오류로 종료"}

        history = list(state.get("node_history", []))
        max_steps = int(state.get("max_steps", 12))
        if len(history) >= max_steps:
            return {
                "route": "finish",
                "route_reason": "최대 실행 단계 도달",
                "remaining_steps": [],
                "errors": [f"최대 실행 단계({max_steps})에 도달하여 종료했습니다."],
            }

        remaining = list(state.get("remaining_steps", []))
        last_node = history[-1] if history else ""
        retry_counts = dict(state.get("retry_counts", {}))
        max_retries = int(state.get("max_retries", 1))

        # 검색 결과를 보고 기존 계획 앞에 키워드 재생성·재검색을 삽입한다.
        if last_node == "search" and not state.get("search_results"):
            attempts = int(retry_counts.get("search", 0))
            if attempts < max_retries:
                retry_counts["search"] = attempts + 1
                return {
                    "route": "keyword",
                    "route_reason": "검색 결과가 없어 키워드를 재생성",
                    "remaining_steps": ["search", *remaining],
                    "retry_counts": retry_counts,
                }
            return {
                "route": "finish",
                "route_reason": "검색 재시도 후에도 결과 없음",
                "remaining_steps": [],
                "retry_counts": retry_counts,
                "errors": ["키워드를 바꿔 재검색했지만 논문을 찾지 못했습니다."],
            }

        research_steps: list[Route] = [
            "keyword",
            "search",
            "download",
            "extract",
            "translate",
            "summarize",
            "rag",
        ]
        # RAG가 문서를 찾으면 해당 문서를 Deep Research에 넘긴다. 문서가 없으면
        # Supervisor가 검색·추출·번역·요약을 거쳐 RAG를 다시 실행한다.
        if last_node == "rag":
            if state.get("sources"):
                return {
                    "route": "deep_research",
                    "route_reason": "RAG 관련 문서를 Deep Research에 전달",
                    "remaining_steps": remaining,
                }

            attempts = int(retry_counts.get("rag_rebuild", 0))
            if attempts < max_retries:
                retry_counts["rag_rebuild"] = attempts + 1
                steps = (
                    ["extract", "translate", "summarize", "rag"]
                    if state.get("paper_ids")
                    else research_steps
                )
                route = steps.pop(0)
                return {
                    "route": route,
                    "route_reason": "RAG 관련 문서가 없어 논문을 재처리",
                    "remaining_steps": steps,
                    "retry_counts": retry_counts,
                }
            return {
                "route": "finish",
                "route_reason": "논문 재처리 후에도 RAG 문서 없음",
                "remaining_steps": [],
                "retry_counts": retry_counts,
                "errors": ["논문을 다시 처리했지만 질문과 관련된 문서를 찾지 못했습니다."],
            }

        # Deep Research가 전달된 문서를 충분히 설명하지 못하면 Supervisor가
        # 추가 논문을 검색하고 전체 처리 흐름을 다시 수행한다.
        if last_node == "deep_research":
            deep_sufficient = (
                state.get("deep_research_status") == "success"
                and bool(state.get("deep_research_sources"))
            )
            if deep_sufficient:
                return {
                    "route": "finish",
                    "route_reason": "Deep Research 설명 완료",
                    "remaining_steps": [],
                }

            attempts = int(retry_counts.get("deep_research", 0))
            if attempts < max_retries:
                retry_counts["deep_research"] = attempts + 1
                steps = list(research_steps)
                route = steps.pop(0)
                return {
                    "route": route,
                    "route_reason": "Deep Research 설명이 부족하여 추가 논문 검색",
                    "remaining_steps": steps,
                    "retry_counts": retry_counts,
                }
            return {
                "route": "finish",
                "route_reason": "추가 검색 후에도 Deep Research 설명 부족",
                "remaining_steps": [],
                "retry_counts": retry_counts,
                "errors": ["추가 논문을 검색했지만 충분한 설명 근거를 찾지 못했습니다."],
            }

        if remaining:
            route = remaining.pop(0)
            return dispatch(
                state,
                route,
                remaining,
                reason="중간 결과를 확인하고 남은 계획을 재검토",
            )
        if history:
            return {"route": "finish", "route_reason": "계획 완료"}
        decision = supervisor_router.decide(state)
        steps: list[Route] = list(decision.steps)
        route = steps.pop(0)
        return dispatch(state, route, steps, reason=decision.reason)

    def finish(state: WorkflowState) -> dict[str, Any]:
        response = _format_response(state)
        return {
            "response": response,
            "messages": [AIMessage(content=response)],
            "node_history": ["finish"],
        }

    builder = StateGraph(WorkflowState)
    builder.add_node("supervisor", supervisor)
    for name, node in graph_nodes.items():
        builder.add_node(name, _guarded(name, node))
    builder.add_node("finish", finish)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        next_route,
        {**{name: name for name in graph_nodes}, "finish": "finish"},
    )
    for name in graph_nodes:
        builder.add_edge(name, "supervisor")
    builder.add_edge("finish", END)

    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=list(interrupt_before),
    )

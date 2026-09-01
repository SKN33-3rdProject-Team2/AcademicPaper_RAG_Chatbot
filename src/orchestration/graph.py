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
    if state.get("summaries"):
        paths = [item.get("markdown_path") for item in state["summaries"] if item.get("markdown_path")]
        return f"논문 {len(state['summaries'])}편의 요약을 완료했습니다." + (
            "\n" + "\n".join(str(path) for path in paths) if paths else ""
        )
    if state.get("translated_paths"):
        return "번역을 완료했습니다.\n" + "\n".join(state["translated_paths"])
    if state.get("extracted_records"):
        return f"논문 {len(state['extracted_records'])}편의 본문 추출을 완료했습니다."
    if state.get("downloaded_paths"):
        return "다운로드를 완료했습니다.\n" + "\n".join(state["downloaded_paths"])
    papers = state.get("search_results") or state.get("library_results") or []
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

        # RAG가 출처를 못 찾거나 스스로 근거 부족을 알리면 Deep Research로 전환한다.
        rag_answer = str(state.get("rag_answer") or state.get("response") or "").casefold()
        insufficient_markers = (
            "근거가 부족",
            "근거를 찾지 못",
            "확인할 수 없",
            "답할 수 없",
            "insufficient evidence",
            "cannot answer",
        )
        rag_insufficient = not state.get("sources") or any(
            marker in rag_answer for marker in insufficient_markers
        )
        if last_node == "rag" and rag_insufficient:
            if "deep_research" not in history:
                return {
                    "route": "deep_research",
                    "route_reason": "RAG 근거가 없어 Deep Research로 전환",
                    "remaining_steps": remaining,
                }
            return {
                "route": "finish",
                "route_reason": "RAG와 Deep Research 모두 완료",
                "remaining_steps": [],
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

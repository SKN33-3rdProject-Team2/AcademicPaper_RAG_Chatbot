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

    @traceable(name="supervisor.plan", run_type="chain")
    def supervisor(state: WorkflowState) -> dict[str, Any]:
        if state.get("errors"):
            return {"route": "finish", "route_reason": "노드 오류로 종료"}
        remaining = list(state.get("remaining_steps", []))
        if remaining:
            route = remaining.pop(0)
            return {"route": route, "remaining_steps": remaining}
        if state.get("node_history"):
            return {"route": "finish", "route_reason": "계획 완료"}
        decision = supervisor_router.decide(state)
        steps: list[Route] = list(decision.steps)
        route = steps.pop(0)
        return {
            "route": route,
            "route_reason": decision.reason,
            "remaining_steps": steps,
        }

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

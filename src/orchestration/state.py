"""Shared state contract for the academic-paper StateGraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


Route = Literal[
    "keyword",
    "search",
    "library",
    "download",
    "extract",
    "translate",
    "summarize",
    "rag",
    "deep_research",
    "finish",
]


class WorkflowState(TypedDict, total=False):
    """Data shared by the supervisor and every tool node.

    Team-owned classes keep their native interfaces. Adapters translate those
    native return values into this state contract.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    route: Route
    route_reason: str
    remaining_steps: list[Route]
    thread_id: str

    keywords: list[str]
    paper_ids: list[str]
    selected_papers: list[dict[str, Any]]
    search_results: list[dict[str, Any]]
    library_results: list[dict[str, Any]]
    downloaded_paths: list[str]
    extracted_records: list[dict[str, Any]]
    translated_paths: list[str]
    summaries: list[dict[str, Any]]

    rag_answer: str
    sources: list[dict[str, Any]]
    response: str

    node_history: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def initial_state(
    query: str,
    *,
    thread_id: str = "default",
    paper_ids: list[str] | None = None,
) -> WorkflowState:
    """Create the minimal valid state accepted by the compiled graph."""

    normalized = query.strip()
    if not normalized:
        raise ValueError("질문을 입력해 주세요.")
    return {
        "messages": [HumanMessage(content=normalized)],
        "query": normalized,
        "thread_id": thread_id,
        "paper_ids": list(paper_ids or []),
        "remaining_steps": [],
        "node_history": [],
        "errors": [],
    }

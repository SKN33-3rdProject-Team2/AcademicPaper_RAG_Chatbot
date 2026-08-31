"""CLI entry point for the LangGraph academic-paper assistant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from orchestration.graph import build_graph
from orchestration.state import initial_state


def run(query: str, *, thread_id: str, paper_ids: list[str] | None = None) -> dict:
    graph = build_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "academic-paper-stategraph",
        "tags": ["academic-paper", "langgraph"],
    }
    return graph.invoke(
        initial_state(query, thread_id=thread_id, paper_ids=paper_ids),
        config=config,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Academic Paper LangGraph chatbot")
    parser.add_argument("query", nargs="?", help="사용자 요청")
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--thread-id", default=f"cli-{uuid4().hex[:8]}")
    args = parser.parse_args()

    query = args.query or input("요청을 입력하세요: ").strip()
    result = run(query, thread_id=args.thread_id, paper_ids=args.paper_id)
    print(result["response"])
    if result.get("sources"):
        print("\n출처:")
        for source in result["sources"]:
            print(f"- [{source['label']}] {source.get('title') or source.get('paper_id')}")
    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())

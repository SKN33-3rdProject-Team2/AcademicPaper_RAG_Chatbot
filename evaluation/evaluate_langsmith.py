"""Create a routing dataset and run a LangSmith experiment."""

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

from langsmith import Client
from langsmith.evaluation import evaluate

from orchestration.evaluation import route_sequence_accuracy
from orchestration.routing import SupervisorRouter
from orchestration.state import initial_state


DATASET_NAME = "academic-paper-supervisor-routing-v1"
ROUTING_EXAMPLES = [
    ({"query": "arXiv에서 graph RAG 논문을 검색해줘"}, {"expected_steps": ["keyword", "search"]}),
    ({"query": "내 서재에 저장된 논문 목록을 보여줘"}, {"expected_steps": ["library"]}),
    ({"query": "paper-001 논문을 번역해줘", "paper_ids": ["paper-001"]}, {"expected_steps": ["extract", "translate"]}),
    ({"query": "paper-001 논문을 요약해줘", "paper_ids": ["paper-001"]}, {"expected_steps": ["extract", "translate", "summarize"]}),
    ({"query": "저장된 요약을 근거와 출처를 붙여 설명해줘"}, {"expected_steps": ["rag"]}),
    ({"query": "로컬 논문들을 비교해서 심층 분석해줘"}, {"expected_steps": ["deep_research"]}),
]


def ensure_dataset(client: Client, dataset_name: str = DATASET_NAME) -> str:
    if client.has_dataset(dataset_name=dataset_name):
        return dataset_name
    client.create_dataset(
        dataset_name,
        description="StateGraph supervisor routing regression set",
    )
    client.create_examples(
        dataset_name=dataset_name,
        examples=[{"inputs": inputs, "outputs": outputs} for inputs, outputs in ROUTING_EXAMPLES],
    )
    return dataset_name


def routing_target(inputs: dict) -> dict:
    router = SupervisorRouter()
    state = initial_state(
        inputs["query"],
        paper_ids=list(inputs.get("paper_ids", [])),
    )
    decision = router.decide(state)
    return {"steps": decision.steps, "reason": decision.reason}


def main() -> int:
    parser = argparse.ArgumentParser(description="LangSmith supervisor routing evaluation")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--prefix", default="academic-paper-routing")
    parser.add_argument("--offline-router", action="store_true")
    args = parser.parse_args()

    client = Client()
    dataset_name = ensure_dataset(client, args.dataset)

    target = routing_target
    if args.offline_router:
        router = SupervisorRouter(use_llm=False)

        def target(inputs: dict) -> dict:
            state = initial_state(inputs["query"], paper_ids=list(inputs.get("paper_ids", [])))
            decision = router.decide(state)
            return {"steps": decision.steps, "reason": decision.reason}

    experiment = evaluate(
        target,
        data=dataset_name,
        evaluators=[route_sequence_accuracy],
        experiment_prefix=f"{args.prefix}-{uuid4().hex[:8]}",
        description="Supervisor route-sequence accuracy",
        max_concurrency=2,
        client=client,
    )
    print(f"LangSmith experiment completed: {experiment.experiment_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

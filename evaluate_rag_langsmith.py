"""Run retrieval and grounded-answer evaluations in LangSmith."""

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

from orchestration.evaluation import (
    LLMJudgeEvaluators,
    citation_precision,
    reciprocal_rank,
    refusal_accuracy,
    retrieval_recall_at_k,
)
from orchestration.rag_chain import SummaryRAGChain
from services.summary_vector_store import ChromaSummaryStore


DATASET_NAME = "academic-paper-rag-grounded-v1"
RAG_CASES = (
    {
        "query": "PACS의 다중 오믹 암 서브타입 예측 방법과 주요 성능을 설명해줘.",
        "paper_id": "2308.10917v1",
    },
    {
        "query": "자연어 지시 기반 로봇 내비게이션 논문의 모델 구조와 Test-New 성능을 설명해줘.",
        "paper_id": "2006.00697v3",
    },
    {
        "query": "Double Multi-Head Attention 화자 검증 모델의 핵심 방법과 실험 결과는 무엇이야?",
        "paper_id": "2007.13199v2",
    },
    {
        "query": "Serialized Multi-Layer Multi-Head Attention 화자 임베딩 방식의 구조와 장점을 설명해줘.",
        "paper_id": "2107.06493v1",
    },
)


def reference_for(store: ChromaSummaryStore, paper_id: str) -> str:
    collection = store._collection()
    result = collection.get(where={"paper_id": paper_id}, include=["documents"])
    documents = [str(item) for item in result.get("documents", []) if item]
    if not documents:
        raise RuntimeError(f"평가용 벡터 문서가 없습니다: {paper_id}")
    return "\n\n".join(documents)


def ensure_dataset(client: Client, dataset_name: str) -> str:
    if client.has_dataset(dataset_name=dataset_name):
        return dataset_name
    store = ChromaSummaryStore()
    examples = []
    for case in RAG_CASES:
        paper_id = case["paper_id"]
        examples.append(
            {
                "inputs": {"query": case["query"]},
                "outputs": {
                    "relevant_source_ids": [paper_id],
                    "reference_answer": reference_for(store, paper_id),
                    "expected_refusal": False,
                },
            }
        )
    examples.append(
        {
            "inputs": {"query": "이 논문 저장소를 근거로 양자 중력의 실험적 증거를 설명해줘."},
            "outputs": {"expected_refusal": True},
        }
    )
    client.create_dataset(
        dataset_name,
        description="Multi-paper retrieval, grounded answer, citation, and refusal evaluation",
    )
    client.create_examples(dataset_name=dataset_name, examples=examples)
    return dataset_name


def rag_target(inputs: dict) -> dict:
    return SummaryRAGChain(top_k=5).invoke(str(inputs["query"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="LangSmith RAG evaluation")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--prefix", default="academic-paper-rag")
    args = parser.parse_args()

    client = Client()
    dataset_name = ensure_dataset(client, args.dataset)
    judges = LLMJudgeEvaluators()
    experiment = evaluate(
        rag_target,
        data=dataset_name,
        evaluators=[
            retrieval_recall_at_k,
            reciprocal_rank,
            citation_precision,
            refusal_accuracy,
            judges.answer_correctness,
            judges.faithfulness,
        ],
        experiment_prefix=f"{args.prefix}-{uuid4().hex[:8]}",
        description="Multi-paper RAG retrieval and grounded-answer quality",
        max_concurrency=1,
        client=client,
    )
    print(f"LangSmith experiment completed: {experiment.experiment_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

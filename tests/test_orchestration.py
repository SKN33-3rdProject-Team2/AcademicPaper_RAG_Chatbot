from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestration.evaluation import (
    citation_precision,
    reciprocal_rank,
    refusal_accuracy,
    retrieval_recall_at_k,
    route_sequence_accuracy,
)
from orchestration.adapters import KeywordNode
from orchestration.graph import build_graph
from orchestration.rag_chain import SummaryRAGChain
from orchestration.routing import SupervisorRouter
from orchestration.routing import SupervisorDecision
from orchestration.state import initial_state


class FakeStore:
    def search(self, query, *, limit=5, paper_id=None):
        return [
            {
                "id": "paper-1:conclusion",
                "document": "이 연구는 검색 증강 생성의 정확도를 개선했다.",
                "metadata": {
                    "paper_id": "paper-1",
                    "title": "RAG Paper",
                    "section": "conclusion",
                },
                "distance": 0.1,
            }
        ]


def fake_nodes():
    return {
        "keyword": lambda state: {"keywords": ["RAG"], "node_history": ["keyword"]},
        "search": lambda state: {
            "search_results": [{"id": "paper-1", "title": "RAG Paper"}],
            "node_history": ["search"],
        },
        "library": lambda state: {"node_history": ["library"]},
        "download": lambda state: {"node_history": ["download"]},
        "extract": lambda state: {"node_history": ["extract"]},
        "translate": lambda state: {"node_history": ["translate"]},
        "summarize": lambda state: {"node_history": ["summarize"]},
        "rag": lambda state: {"response": "RAG answer", "node_history": ["rag"]},
        "deep_research": lambda state: {"response": "Deep answer", "node_history": ["deep_research"]},
    }


class StateGraphTest(unittest.TestCase):
    def test_search_plan_executes_in_order(self):
        graph = build_graph(router=SupervisorRouter(use_llm=False), nodes=fake_nodes())
        result = graph.invoke(
            initial_state("arxiv에서 RAG 논문 검색해줘"),
            config={"configurable": {"thread_id": "test-search"}},
        )
        self.assertEqual(result["node_history"], ["keyword", "search", "finish"])
        self.assertIn("RAG Paper", result["response"])

    def test_summary_plan_keeps_extract_translate_order(self):
        router = SupervisorRouter(use_llm=False)
        decision = router.decide(initial_state("paper-1 논문을 요약해줘", paper_ids=["paper-1"]))
        self.assertEqual(decision.steps, ["extract", "translate", "summarize"])

    def test_explicit_routes_do_not_overplan(self):
        router = SupervisorRouter(use_llm=True)
        cases = [
            ("내 서재에 저장된 논문 목록을 보여줘", ["library"]),
            ("저장된 요약을 근거와 출처를 붙여 설명해줘", ["rag"]),
            ("로컬 논문들을 비교해서 심층 분석해줘", ["deep_research"]),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(router.decide(initial_state(query)).steps, expected)

    def test_empty_search_rebuilds_keywords_and_retries(self):
        search_calls = 0

        def search_node(state):
            nonlocal search_calls
            search_calls += 1
            papers = [] if search_calls == 1 else [{"id": "paper-2", "title": "Retry Paper"}]
            return {"search_results": papers, "node_history": ["search"]}

        nodes = fake_nodes()
        nodes["search"] = search_node
        graph = build_graph(router=SupervisorRouter(use_llm=False), nodes=nodes)
        result = graph.invoke(
            initial_state("arxiv에서 agent 논문 검색해줘"),
            config={"configurable": {"thread_id": "test-search-retry"}},
        )
        self.assertEqual(
            result["node_history"],
            ["keyword", "search", "keyword", "search", "finish"],
        )
        self.assertEqual(result["retry_counts"]["search"], 1)
        self.assertIn("Retry Paper", result["response"])

    def test_keyword_retry_requests_alternative_terms(self):
        prompts = []

        class FakeKeywordTool:
            def generate_keywords(self, prompt):
                prompts.append(prompt)
                return {"keywords": ["alternative RAG"]}

        node = KeywordNode(factory=FakeKeywordTool)
        result = node(
            {
                "query": "RAG 논문",
                "keywords": ["retrieval augmented generation"],
                "retry_counts": {"search": 1},
            }
        )
        self.assertEqual(result["keywords"], ["alternative RAG"])
        self.assertIn("retrieval augmented generation", prompts[0])
        self.assertIn("겹치지 않는 대체 학술 용어", prompts[0])

    def test_rag_without_sources_falls_back_to_deep_research(self):
        nodes = fake_nodes()
        nodes["rag"] = lambda state: {
            "response": "근거를 찾지 못했습니다.",
            "sources": [],
            "node_history": ["rag"],
        }
        graph = build_graph(router=SupervisorRouter(use_llm=False), nodes=nodes)
        result = graph.invoke(
            initial_state("저장된 요약을 근거와 출처를 붙여 설명해줘"),
            config={"configurable": {"thread_id": "test-rag-fallback"}},
        )
        self.assertEqual(result["node_history"], ["rag", "deep_research", "finish"])
        self.assertEqual(result["response"], "Deep answer")

    def test_rag_refusal_with_sources_also_falls_back(self):
        nodes = fake_nodes()
        nodes["rag"] = lambda state: {
            "rag_answer": "저장된 근거가 부족하여 답할 수 없습니다.",
            "response": "저장된 근거가 부족하여 답할 수 없습니다.",
            "sources": [{"id": "weak-source"}],
            "node_history": ["rag"],
        }
        graph = build_graph(router=SupervisorRouter(use_llm=False), nodes=nodes)
        result = graph.invoke(
            initial_state("저장된 요약을 근거와 출처를 붙여 설명해줘"),
            config={"configurable": {"thread_id": "test-rag-refusal-fallback"}},
        )
        self.assertEqual(result["node_history"], ["rag", "deep_research", "finish"])

    def test_missing_translation_input_inserts_extract_node(self):
        class TranslateOnlyRouter:
            def decide(self, state):
                return SupervisorDecision(steps=["translate"], reason="번역만 계획")

        nodes = fake_nodes()
        nodes["extract"] = lambda state: {
            "extracted_records": [{"id": "paper-1", "content": "paper body"}],
            "node_history": ["extract"],
        }
        nodes["translate"] = lambda state: {
            "translated_paths": ["paper-1-ko.md"],
            "node_history": ["translate"],
        }
        graph = build_graph(router=TranslateOnlyRouter(), nodes=nodes)
        result = graph.invoke(
            initial_state("번역", paper_ids=["paper-1"]),
            config={"configurable": {"thread_id": "test-translation-prerequisite"}},
        )
        self.assertEqual(result["node_history"], ["extract", "translate", "finish"])

    def test_retry_limit_stops_an_infinite_search_loop(self):
        nodes = fake_nodes()
        nodes["search"] = lambda state: {"search_results": [], "node_history": ["search"]}
        graph = build_graph(router=SupervisorRouter(use_llm=False), nodes=nodes)
        result = graph.invoke(
            initial_state("arxiv에서 없는 논문 검색해줘", max_retries=1),
            config={"configurable": {"thread_id": "test-search-limit"}},
        )
        self.assertEqual(
            result["node_history"],
            ["keyword", "search", "keyword", "search", "finish"],
        )
        self.assertTrue(result["errors"])
        self.assertIn("찾지 못했습니다", result["response"])


class RAGChainTest(unittest.TestCase):
    def test_returns_answer_and_sources(self):
        fake_llm = RunnableLambda(lambda prompt: AIMessage(content="정확도가 개선되었습니다 [S1]."))
        chain = SummaryRAGChain(store_factory=FakeStore, llm=fake_llm)
        result = chain.invoke("결과가 뭐야?")
        self.assertEqual(result["sources"][0]["label"], "S1")
        self.assertIn("[S1]", result["answer"])


class EvaluatorTest(unittest.TestCase):
    def test_deterministic_metrics(self):
        route = route_sequence_accuracy({}, {"steps": ["rag"]}, {"expected_steps": ["rag"]})
        recall = retrieval_recall_at_k(
            {}, {"sources": [{"id": "p1"}]}, {"relevant_source_ids": ["p1", "p2"]}
        )
        mrr = reciprocal_rank(
            {}, {"sources": [{"id": "x"}, {"id": "p1"}]}, {"relevant_source_ids": ["p1"]}
        )
        citation = citation_precision(
            {}, {"answer": "근거 [S1] [S9]", "sources": [{"label": "S1"}]}, {}
        )
        self.assertEqual(route.score, 1.0)
        self.assertEqual(recall.score, 0.5)
        self.assertEqual(mrr.score, 0.5)
        self.assertEqual(citation.score, 0.5)

    def test_not_applicable_metrics_do_not_inflate_score(self):
        result = retrieval_recall_at_k({}, {"sources": []}, {})
        self.assertIsNone(result.score)

    def test_retrieval_metrics_match_paper_id_and_document_id(self):
        outputs = {
            "sources": [
                {"id": "paper-1:methodology", "paper_id": "paper-1"},
            ]
        }
        reference = {"relevant_source_ids": ["paper-1"]}
        self.assertEqual(retrieval_recall_at_k({}, outputs, reference).score, 1.0)
        self.assertEqual(reciprocal_rank({}, outputs, reference).score, 1.0)

    def test_refusal_accuracy(self):
        result = refusal_accuracy(
            {},
            {"answer": "저장된 근거가 부족하여 답할 수 없습니다."},
            {"expected_refusal": True},
        )
        self.assertEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()

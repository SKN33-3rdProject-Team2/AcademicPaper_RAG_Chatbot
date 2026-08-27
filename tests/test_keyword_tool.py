"""Ollama 키워드 생성 Tool의 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 직접 실행(`python tests/test_keyword_tool.py`)할 때도 프로젝트 모듈을 찾도록 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.ollama_service import OllamaServiceError
from tools.keyword_tool import KeywordToolError, generate_arxiv_keywords


# 실제 Ollama 키워드 생성을 확인할 사용자 입력. 원하는 주제로 이 값만 바꾼다.
USER_PROMPT = "LLM을 활용한 논문들이 뭐가 있는지 찾아줘"


class KeywordToolTest(unittest.TestCase):
    @patch("tools.keyword_tool.generate")
    def test_generates_english_keywords_from_korean_topic(self, mock_generate):
        """한글 연구 주제에 대해 정리된 영문 키워드 목록을 반환한다."""
        mock_generate.return_value = (
            '{"keywords": ["multi-agent collaboration", "cooperative LLM agents", '
            '"agent communication", "distributed AI systems", "LLM orchestration", '
            '"collaborative artificial intelligence", "Multi-Agent Collaboration"]}'
        )

        result = generate_arxiv_keywords.invoke({"user_query": USER_PROMPT})

        self.assertEqual(
            result["keywords"],
            [
                "multi-agent collaboration",
                "cooperative LLM agents",
                "agent communication",
                "distributed AI systems",
                "LLM orchestration",
                "collaborative artificial intelligence",
            ],
        )
        prompt = mock_generate.call_args.args[0]
        self.assertIn(USER_PROMPT, prompt)
        self.assertEqual(mock_generate.call_args.kwargs["response_format"], "json")

    @patch("tools.keyword_tool.generate", return_value='{"error": "Please enter a clear research topic."}')
    def test_returns_model_error_for_unreadable_input(self, _mock_generate):
        """모델이 오류 JSON을 반환하면 Tool도 그 오류를 호출자에게 알린다."""
        with self.assertRaisesRegex(KeywordToolError, "Please enter a clear research topic"):
            generate_arxiv_keywords.invoke({"user_query": "@@@"})

    def test_rejects_empty_input_without_calling_ollama(self):
        """빈 입력은 Ollama 호출 전에 오류로 반환한다."""
        with self.assertRaisesRegex(KeywordToolError, "검색할 연구 주제를 입력"):
            generate_arxiv_keywords.invoke({"user_query": "   "})

    @patch("tools.keyword_tool.generate", return_value="not JSON")
    def test_converts_invalid_model_response_to_tool_error(self, _mock_generate):
        """JSON이 아닌 모델 응답은 호출자에게 명확한 Tool 오류로 전달한다."""
        with self.assertRaisesRegex(KeywordToolError, "Ollama 응답"):
            generate_arxiv_keywords.invoke({"user_query": "RAG 평가"})

    @patch("tools.keyword_tool.generate", side_effect=OllamaServiceError("connection refused"))
    def test_converts_ollama_connection_error_to_tool_error(self, _mock_generate):
        """Ollama 연결 실패는 Tool 전용 오류로 변환한다."""
        with self.assertRaisesRegex(KeywordToolError, "키워드를 생성하지 못했습니다"):
            generate_arxiv_keywords.invoke({"user_query": "RAG 평가"})

    def test_generates_keywords_with_user_prompt(self):
        """테스트 파일 상단의 사용자 입력으로 실제 Ollama 키워드를 생성한다."""
        result = generate_arxiv_keywords.invoke({"user_query": USER_PROMPT})

        self.assertEqual(len(result["keywords"]), 6)
        self.assertTrue(all(isinstance(keyword, str) and keyword for keyword in result["keywords"]))
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print("\nGenerated keywords:")
        for index, keyword in enumerate(result["keywords"], start=1):
            print(f"{index}. {keyword}")


if __name__ == "__main__":
    unittest.main()

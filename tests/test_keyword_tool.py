"""Ollama 키워드 생성 Tool의 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 직접 실행(`python tests/test_keyword_tool.py`)할 때도 프로젝트 모듈을 찾도록 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.keyword_tool import KeywordTool


# 실제 Ollama 키워드 생성을 확인할 사용자 입력. 원하는 주제로 이 값만 바꾼다.
USER_PROMPT = "LLM을 활용한 논문들이 뭐가 있는지 찾아줘"


class KeywordToolTest(unittest.TestCase):
    def test_generates_keywords_with_user_prompt(self):
        """테스트 파일 상단의 사용자 입력으로 실제 Ollama 키워드를 생성한다."""
        result = KeywordTool().generate_keywords(USER_PROMPT)

        self.assertEqual(len(result["keywords"]), 6)
        self.assertTrue(all(isinstance(keyword, str) and keyword for keyword in result["keywords"]))
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print("\nGenerated keywords:")
        for index, keyword in enumerate(result["keywords"], start=1):
            print(f"{index}. {keyword}")


if __name__ == "__main__":
    unittest.main()

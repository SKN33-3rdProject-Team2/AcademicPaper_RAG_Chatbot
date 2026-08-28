"""Ollama를 이용해 arXiv 검색 키워드를 생성하는 Tool."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import tool

# ---------------------------------------------------------------------
# 🚨 [모듈 임포트 경로 설정]
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from services.ollama_service import OllamaServiceError, generate
# ---------------------------------------------------------------------


class KeywordToolError(RuntimeError):
    """키워드 Tool이 유효한 검색 키워드를 만들지 못했을 때 발생한다."""


def _clean_keywords(values: object, max_keywords: int = 6) -> list[str]:
    if not isinstance(values, list):
        return []
    keywords: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split()).strip('"')
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            keywords.append(cleaned)
    return keywords[:max_keywords]


class KeywordTool:
    """Ollama로 사용자 연구 주제를 arXiv 검색 키워드로 변환한다."""

    def __init__(self, generator: Callable[..., str] | None = None) -> None:
        self._generator = generator or generate

    def generate_keywords(self, user_query: str) -> dict[str, list[str]]:
        """사용자 연구 주제를 영문 학술 검색 키워드 6개로 변환한다."""
        raw_query = user_query.strip()
        if not raw_query:
            raise KeywordToolError("검색할 연구 주제를 입력해 주세요.")

        # 💡 [방어 로직] "LLM", "AI" 등 너무 짧은 단어는 Ollama가 거부하므로 자동으로 의미 확장 보완
        if len(raw_query) <= 3:
            raw_query = f"{raw_query} technology and models"

        prompt = f"""You are an expert in academic literature search for arXiv.
Your task is to convert the user's research topic into exactly 6 concise, highly specific English academic search terms or technical keywords.

[Rules]
1. DO NOT use vague conversational phrases, adjectives, or opinions.
2. Use precise computer science or technical terminology (e.g., instead of short words, expand them into standard academic terms like 'Large Language Models', 'Transformer', 'Neural Networks').
3. Each phrase must be short (1 to 3 words) and ideal for boolean search engines like arXiv.
4. Return JSON only in this exact shape:
{{"keywords": ["keyword one", "keyword two", "keyword three", "keyword four", "keyword five", "keyword six"]}}

User topic: {raw_query}"""

        try:
            response = self._generator(prompt, response_format="json")
            payload = json.loads(response)
        except Exception:
            # 💡 [Fallback 방어] Ollama 연결 실패나 파싱 에러 시 절대 뻗지 않고 기본 학술 키워드 제공
            base_kw = user_query.strip()
            return {"keywords": [base_kw, f"{base_kw} models", f"{base_kw} architecture", "Deep Learning", "Neural Networks", "Transformer"]}

        if not isinstance(payload, dict):
            return {"keywords": [user_query, "Large Language Models", "Deep Learning", "Neural Networks", "Transformer", "Artificial Intelligence"]}

        error_message = payload.get("error")
        if isinstance(error_message, str) and error_message.strip():
            # 에러가 나도 기본 키워드로 자동 우회 처리
            base_kw = user_query.strip()
            return {"keywords": [base_kw, f"{base_kw} models", "Large Language Models", "Deep Learning", "Neural Networks", "Transformer"]}

        keywords = _clean_keywords(payload.get("keywords"))

        # 만약 6개가 안 채워졌다면 부족한 만큼 기본 키워드로 채워넣기
        if len(keywords) < 6:
            defaults = [user_query.strip(), "Large Language Models", "Deep Learning", "Neural Networks", "Transformer", "Artificial Intelligence"]
            for d in defaults:
                if d.casefold() not in [k.casefold() for k in keywords]:
                    keywords.append(d)
                if len(keywords) == 6:
                    break

        return {"keywords": keywords[:6]}


@tool
def generate_arxiv_keywords(user_query: str) -> dict[str, list[str]]:
    """사용자 연구 주제를 arXiv 검색용 영문 학술 키워드로 변환한다."""
    return KeywordTool().generate_keywords(user_query)
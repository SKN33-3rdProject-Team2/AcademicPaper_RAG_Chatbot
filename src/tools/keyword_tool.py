"""Ollama를 이용해 arXiv 검색 키워드를 생성하는 Tool."""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.tools import tool
from log import AppLogger, LogCode

from services.generation_options import resolve_float, resolve_positive_int
from services.model_config_service import load_task_config
from services.ollama_service import generate

logger = AppLogger(__name__)
KEYWORD_CONFIG = load_task_config("keyword")


class KeywordToolError(RuntimeError):
    """키워드 Tool이 유효한 검색 키워드를 만들지 못했을 때 발생한다."""


def _clean_keywords(values: object) -> list[str]:
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
    return keywords


class KeywordTool:
    """Ollama로 사용자 연구 주제를 arXiv 검색 키워드로 변환한다."""

    def __init__(
        self,
        generator: Callable[..., str] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self._generator = generator or generate
        self._model = str(KEYWORD_CONFIG["model"])
        self._temperature = resolve_float(
            "temperature",
            KEYWORD_CONFIG["temperature"]
            if temperature is None
            else temperature,
            minimum=0.0,
            maximum=2.0,
        )
        self._max_tokens = resolve_positive_int(
            "max_tokens",
            KEYWORD_CONFIG["max_tokens"]
            if max_tokens is None
            else max_tokens,
        )
        self._timeout = resolve_float(
            "timeout",
            KEYWORD_CONFIG["timeout"] if timeout is None else timeout,
            minimum=0.1,
        )

    def generate_keywords(self, user_query: str) -> dict[str, list[str]]:
        """사용자 연구 주제를 영문 학술 검색 키워드 6개로 변환한다."""
        raw_query = user_query.strip()
        if not raw_query:
            logger.log(LogCode.KEYWORD_GENERATION_REJECTED, reason="empty_query")
            raise KeywordToolError("검색할 연구 주제를 입력해 주세요.")

        logger.log(
            LogCode.KEYWORD_GENERATION_STARTED,
            query=raw_query,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )

        prompt = f"""You are an expert in academic literature search for arXiv.
Your task is to convert the user's research topic into exactly 6 concise, highly specific English academic search terms or technical keywords.

[Rules]
1. Every keyword must be directly and clearly related to the user's original research topic.
2. Return exactly 6 unique keywords. Never return fewer or more than 6.
3. DO NOT use vague conversational phrases, adjectives, opinions, or unrelated general terms.
4. Use precise academic or technical terminology appropriate for arXiv.
5. Each keyword must be a short English phrase of 1 to 3 words and suitable for an arXiv search.
6. Return JSON only in this exact shape, without explanations or additional fields:
{{"keywords": ["keyword one", "keyword two", "keyword three", "keyword four", "keyword five", "keyword six"]}}

User topic: {raw_query}"""

        response = self._generator(
            prompt,
            model=self._model,
            response_format="json",
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )
        payload = json.loads(response)

        if not isinstance(payload, dict):
            logger.log(
                LogCode.KEYWORD_GENERATION_FAILED,
                query=raw_query,
                reason="response_is_not_object",
                response_type=type(payload).__name__,
            )
            raise KeywordToolError("Ollama 응답이 JSON 객체 형식이 아닙니다.")

        error_message = payload.get("error")
        if isinstance(error_message, str) and error_message.strip():
            logger.log(
                LogCode.KEYWORD_GENERATION_FAILED,
                query=raw_query,
                reason="ollama_response_error",
                error=error_message.strip(),
            )
            raise KeywordToolError(error_message.strip())

        keywords = _clean_keywords(payload.get("keywords"))

        if len(keywords) != 6:
            logger.log(
                LogCode.KEYWORD_GENERATION_FAILED,
                query=raw_query,
                reason="invalid_keyword_count",
                keyword_count=len(keywords),
            )
            raise KeywordToolError(
                f"Ollama는 중복 없는 유효한 키워드 6개를 반환해야 합니다: {len(keywords)}개 반환됨"
            )

        logger.log(
            LogCode.KEYWORD_GENERATION_SUCCEEDED,
            query=raw_query,
            keywords=keywords,
            keyword_count=len(keywords),
        )
        return {"keywords": keywords}


@tool
def generate_arxiv_keywords(user_query: str) -> dict[str, list[str]]:
    """사용자 연구 주제를 arXiv 검색용 영문 학술 키워드로 변환한다."""
    return KeywordTool().generate_keywords(user_query)

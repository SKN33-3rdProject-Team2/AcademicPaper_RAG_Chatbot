"""Ollama를 이용해 arXiv 검색 키워드를 생성하는 Tool."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from services.ollama_service import OllamaServiceError, generate


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


@tool
def generate_arxiv_keywords(user_query: str) -> dict[str, list[str]]:
    """사용자 연구 주제를 arXiv 검색용 영문 학술 키워드로 변환한다."""
    if not user_query.strip():
        raise KeywordToolError("검색할 연구 주제를 입력해 주세요.")

    prompt = f"""You create arXiv search keywords.
Convert a clear research topic into exactly 6 concise English academic search phrases.
Return JSON only. For a valid topic, return exactly this shape:
{{"keywords": ["phrase one", "phrase two"]}}
If the user input is empty, unreadable, meaningless, or has no identifiable research topic,
return exactly this shape instead:
{{"error": "Please enter a clear research topic."}}
Do not include explanations, Korean, Chinese, arXiv query syntax, or duplicate phrases.

User topic: {user_query}"""
    try:
        response = generate(prompt, response_format="json")
        payload = json.loads(response)
    except (OllamaServiceError, json.JSONDecodeError) as exc:
        raise KeywordToolError("키워드를 생성하지 못했습니다. Ollama 응답을 확인해 주세요.") from exc

    if not isinstance(payload, dict):
        raise KeywordToolError("Ollama 응답 형식을 읽을 수 없습니다.")
    error_message = payload.get("error")
    if isinstance(error_message, str) and error_message.strip():
        raise KeywordToolError(error_message.strip())

    keywords = _clean_keywords(payload.get("keywords"))
    if len(keywords) != 6:
        raise KeywordToolError("Ollama가 6개의 검색 키워드를 생성하지 못했습니다.")
    return {"keywords": keywords}

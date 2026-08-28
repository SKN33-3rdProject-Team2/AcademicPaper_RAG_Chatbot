"""로그 코드별 레벨, 이벤트 이름, 사용자 메시지 정의."""

from __future__ import annotations

from .log_codes import LogCode


LOG_MESSAGES: dict[LogCode, dict[str, str]] = {
    LogCode.OLLAMA_CONNECTION_CHECKED: {
        "level": "INFO",
        "event": "ollama_connection_checked",
        "message": "Ollama 서버 연결 및 모델 설치 상태를 확인했습니다.",
    },
    LogCode.OLLAMA_GENERATION_SUCCEEDED: {
        "level": "INFO",
        "event": "ollama_generation_succeeded",
        "message": "Ollama가 응답 생성을 완료했습니다.",
    },
    LogCode.OLLAMA_CONNECTION_FAILED: {
        "level": "ERROR",
        "event": "ollama_connection_failed",
        "message": "Ollama 서버에 연결하지 못했습니다.",
    },
    LogCode.OLLAMA_GENERATION_FAILED: {
        "level": "ERROR",
        "event": "ollama_generation_failed",
        "message": "Ollama 응답 생성 요청에 실패했습니다.",
    },
    LogCode.OLLAMA_CONFIG_LOAD_FAILED: {
        "level": "ERROR",
        "event": "ollama_config_load_failed",
        "message": "Ollama 설정 파일을 불러오지 못했습니다.",
    },
    LogCode.KEYWORD_GENERATION_STARTED: {
        "level": "INFO",
        "event": "keyword_generation_started",
        "message": "학술 검색 키워드 생성을 시작합니다.",
    },
    LogCode.KEYWORD_GENERATION_SUCCEEDED: {
        "level": "INFO",
        "event": "keyword_generation_succeeded",
        "message": "학술 검색 키워드 6개를 생성했습니다.",
    },
    LogCode.KEYWORD_GENERATION_FALLBACK: {
        "level": "WARNING",
        "event": "keyword_generation_fallback",
        "message": "키워드 생성 또는 응답 처리 문제로 기본 키워드를 사용합니다.",
    },
    LogCode.KEYWORD_GENERATION_REJECTED: {
        "level": "WARNING",
        "event": "keyword_generation_rejected",
        "message": "검색어가 비어 있어 키워드 생성을 시작하지 않았습니다.",
    },
    LogCode.KEYWORD_GENERATION_FAILED: {
        "level": "ERROR",
        "event": "keyword_generation_failed",
        "message": "Ollama가 중복 없는 유효한 학술 검색 키워드 6개를 반환하지 않았습니다.",
    },
}

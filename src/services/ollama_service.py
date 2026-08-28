"""로컬 Ollama API 호출만 담당하는 서비스."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import yaml

from log import AppLogger, LogCode
from . import OLLAMA_CONFIG_PATH

logger = AppLogger(__name__)


def _load_ollama_config() -> dict[str, str]:
    """팀 공통 Ollama YAML 설정에서 model과 host를 읽는다."""
    if not OLLAMA_CONFIG_PATH.is_file():
        logger.log(
            LogCode.OLLAMA_CONFIG_LOAD_FAILED,
            reason="config_file_not_found",
            config_path=OLLAMA_CONFIG_PATH,
        )
        raise RuntimeError(f"Ollama 설정 파일을 찾을 수 없습니다: {OLLAMA_CONFIG_PATH}")

    try:
        with OLLAMA_CONFIG_PATH.open(encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        logger.log(
            LogCode.OLLAMA_CONFIG_LOAD_FAILED,
            reason="config_parse_failed",
            config_path=OLLAMA_CONFIG_PATH,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise RuntimeError(
            "Ollama 설정 파일을 읽거나 파싱하지 못했습니다."
        ) from exc

    config: dict[str, str] = {}
    if isinstance(yaml_data, dict):
        for key, value in yaml_data.items():
            if key in {"model", "host"} and value:
                config[key] = str(value).strip()

    if "model" not in config or "host" not in config:
        logger.log(
            LogCode.OLLAMA_CONFIG_LOAD_FAILED,
            reason="required_fields_missing",
            config_path=OLLAMA_CONFIG_PATH,
            missing_fields=[key for key in ("model", "host") if key not in config],
        )
        raise RuntimeError("ollama_config.yaml에는 model과 host 설정이 필요합니다.")
    return config


OLLAMA_CONFIG = _load_ollama_config()


class OllamaServiceError(RuntimeError):
    """Ollama 서버 요청을 완료하지 못했을 때 발생한다."""


def check_connection(timeout: int = 3) -> bool:
    """Ollama 서버와 설정된 모델의 사용 가능 여부를 확인하고 로그로 남긴다."""
    host = OLLAMA_CONFIG["host"].rstrip("/")
    model = OLLAMA_CONFIG["model"]
    started_at = time.perf_counter()

    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise TypeError("Ollama 모델 목록 응답이 JSON 객체가 아닙니다.")
        models = body.get("models")
        if not isinstance(models, list):
            raise TypeError("Ollama 모델 목록 응답에 models 배열이 없습니다.")
        available_models = {
            item.get("name")
            for item in models
            if isinstance(item, dict)
        }
        connected = model in available_models
        logger.log(
            LogCode.OLLAMA_CONNECTION_CHECKED,
            connected=connected,
            host=host,
            model=model,
            model_available=connected,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return connected
    except (OSError, urllib.error.URLError, TimeoutError, ValueError, TypeError) as exc:
        logger.log(
            LogCode.OLLAMA_CONNECTION_FAILED,
            host=host,
            model=model,
            reason="connection_or_response_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return False


def generate(prompt: str, *, response_format: str | None = None, timeout: int = 25) -> str:
    """Ollama generate API를 호출하고 모델의 텍스트 응답만 반환한다."""
    host = OLLAMA_CONFIG["host"].rstrip("/")
    started_at = time.perf_counter()
    payload: dict[str, object] = {
        "model": OLLAMA_CONFIG["model"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 160},
    }
    if response_format:
        payload["format"] = response_format

    request = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        logger.log(
            LogCode.OLLAMA_GENERATION_FAILED,
            connected=False,
            host=host,
            model=OLLAMA_CONFIG["model"],
            reason="request_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        raise OllamaServiceError(
            "Ollama에 연결하지 못했습니다. Ollama 실행 상태와 모델 설치 상태를 확인해 주세요."
        ) from exc

    try:
        body = json.loads(response_body.decode("utf-8"))
        if not isinstance(body, dict):
            raise TypeError("Ollama 생성 응답이 JSON 객체가 아닙니다.")
        result = body["response"]
        if not isinstance(result, str):
            raise TypeError("Ollama 응답에 텍스트가 없습니다.")
        logger.log(
            LogCode.OLLAMA_GENERATION_SUCCEEDED,
            connected=True,
            host=host,
            model=OLLAMA_CONFIG["model"],
            response_format=response_format,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return result
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.log(
            LogCode.OLLAMA_GENERATION_FAILED,
            connected=True,
            host=host,
            model=OLLAMA_CONFIG["model"],
            reason="invalid_response",
            error_type=type(exc).__name__,
            error=str(exc),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        raise OllamaServiceError(
            "Ollama 응답 형식이 올바르지 않습니다."
        ) from exc

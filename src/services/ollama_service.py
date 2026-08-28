"""로컬 Ollama API 호출만 담당하는 서비스."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import yaml

# 파일 위치: src/services/ollama_service.py -> 부모의 부모는 src/
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Log import AppLogger, LogCode

OLLAMA_CONFIG_PATH = SRC_DIR / "config" / "ollama_config.yaml"


def _load_ollama_config() -> dict[str, str]:
    """팀 공통 Ollama YAML 설정에서 model과 host를 읽는다."""
    if not OLLAMA_CONFIG_PATH.is_file():
        raise RuntimeError(f"Ollama 설정 파일을 찾을 수 없습니다: {OLLAMA_CONFIG_PATH}")

    config: dict[str, str] = {}

    # yaml 모듈을 이용해 안전하게 로드하도록 개선
    try:
        with open(OLLAMA_CONFIG_PATH, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            if isinstance(yaml_data, dict):
                for k, v in yaml_data.items():
                    if k in {"model", "host"} and v:
                        config[k] = str(v).strip()
    except Exception:
        # 기존 라인 파싱 방식 Fallback
        for line in OLLAMA_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            value = value.strip().strip('"').strip("'")
            if key in {"model", "host"} and value:
                config[key] = value

    if "model" not in config or "host" not in config:
        raise RuntimeError("ollama_config.yaml에는 model과 host 설정이 필요합니다.")
    return config


OLLAMA_CONFIG = _load_ollama_config()
logger = AppLogger(__name__)


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
        available_models = {
            item.get("name")
            for item in body.get("models", [])
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
            body = json.loads(response.read().decode("utf-8"))
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
    except (OSError, urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError) as exc:
        logger.log(
            LogCode.OLLAMA_GENERATION_FAILED,
            connected=False,
            host=host,
            model=OLLAMA_CONFIG["model"],
            error_type=type(exc).__name__,
            error=str(exc),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        raise OllamaServiceError(
            "Ollama에 연결하지 못했습니다. Ollama 실행 상태와 모델 설치 상태를 확인해 주세요."
        ) from exc

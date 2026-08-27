"""로컬 Ollama API 호출만 담당하는 서비스."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path


OLLAMA_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ollama_config.yaml"


def _load_ollama_config() -> dict[str, str]:
    """팀 공통 Ollama YAML 설정에서 model과 host를 읽는다."""
    if not OLLAMA_CONFIG_PATH.is_file():
        raise RuntimeError(f"Ollama 설정 파일을 찾을 수 없습니다: {OLLAMA_CONFIG_PATH}")

    config: dict[str, str] = {}
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


class OllamaServiceError(RuntimeError):
    """Ollama 서버 요청을 완료하지 못했을 때 발생한다."""


def generate(prompt: str, *, response_format: str | None = None, timeout: int = 25) -> str:
    """Ollama generate API를 호출하고 모델의 텍스트 응답만 반환한다."""
    host = OLLAMA_CONFIG["host"].rstrip("/")
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
        return result
    except (OSError, urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError) as exc:
        raise OllamaServiceError(
            "Ollama에 연결하지 못했습니다. Ollama 실행 상태와 모델 설치 상태를 확인해 주세요."
        ) from exc

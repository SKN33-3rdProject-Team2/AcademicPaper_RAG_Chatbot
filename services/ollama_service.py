"""로컬 Ollama API 호출만 담당하는 서비스."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"

def _load_project_env() -> None:
    """외부 패키지 없이 프로젝트 루트의 .env 설정을 읽는다."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_load_project_env()


class OllamaServiceError(RuntimeError):
    """Ollama 서버 요청을 완료하지 못했을 때 발생한다."""


def generate(prompt: str, *, response_format: str | None = None, timeout: int = 25) -> str:
    """Ollama generate API를 호출하고 모델의 텍스트 응답만 반환한다."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    payload: dict[str, object] = {
        "model": os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
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

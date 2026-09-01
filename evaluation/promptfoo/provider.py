"""Promptfoo Python provider for the project Supervisor chatbot."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
load_dotenv(PROJECT_ROOT / ".env", override=False)

from feature.supervisor_chatbot import SupervisorChatbot


_CHATBOT: SupervisorChatbot | None = None


def _chatbot() -> SupervisorChatbot:
    global _CHATBOT
    if _CHATBOT is None:
        _CHATBOT = SupervisorChatbot()
    return _CHATBOT


def _prompt_text(prompt: object) -> str:
    if not isinstance(prompt, str):
        return str(prompt)
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if isinstance(payload, list):
        messages = [
            str(item.get("content") or "")
            for item in payload
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        if messages:
            return messages[-1]
    return prompt


def call_api(prompt: str, options: dict, context: dict) -> dict[str, Any]:
    """Run one isolated Promptfoo case through the real LangGraph chatbot."""

    query = _prompt_text(prompt).strip()
    variables = context.get("vars", {}) if isinstance(context, dict) else {}
    paper_ids = variables.get("paper_ids") or []
    if isinstance(paper_ids, str):
        paper_ids = [value.strip() for value in paper_ids.split(",") if value.strip()]
    try:
        result = _chatbot().invoke(
            query,
            thread_id=f"promptfoo-{uuid4().hex[:10]}",
            paper_ids=list(paper_ids),
        )
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}

    errors = [str(value) for value in result.get("errors", []) if str(value).strip()]
    if errors:
        return {"error": " | ".join(errors)}
    return {
        "output": str(result.get("response") or ""),
        "metadata": {
            "steps": list(result.get("node_history", [])),
            "source_count": len(result.get("sources", [])),
        },
    }


__all__ = ["call_api"]

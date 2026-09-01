"""Generate Promptfoo cases from the checked-in v2 evaluation dataset."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset import build_examples


def _case(example: dict[str, Any]) -> dict[str, Any]:
    inputs = example["inputs"]
    expected = example["outputs"]
    variables = {
        "query": str(inputs["query"]),
        "suite": str(inputs["suite"]),
        "paper_ids": list(inputs.get("paper_ids", [])),
        "required_terms": list(expected.get("required_terms", [])),
        "expected_refusal": expected.get("expected_refusal"),
    }
    return {
        "description": str(inputs["case_id"]),
        "vars": variables,
    }


def generate_tests(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return a small, representative live regression set by default."""

    settings = config or {}
    max_cases = int(settings.get("max_cases", 10))
    examples = [
        *build_examples("deep_research"),
        *[
            example
            for example in build_examples("rag")
            if example["outputs"].get("expected_refusal") is True
        ],
        *build_examples("pipeline")[:2],
    ]
    return [_case(example) for example in examples[:max_cases]]


__all__ = ["generate_tests"]

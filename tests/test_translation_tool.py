"""TranslateTool의 Markdown 파일 입력 테스트."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from tests import SRC_DIR  # noqa: F401 - src 경로를 sys.path에 등록한다.

from services import PROJECT_ROOT
from tools.translation_tool import TranslateTool


TEST_MARKDOWN_PATH = PROJECT_ROOT / "data" / "paper_extract" / "2312.04649v1.md"
TRANSLATION_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "translations"


class TranslationToolTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_MODEL_INTEGRATION_TESTS") == "1",
        "RUN_MODEL_INTEGRATION_TESTS=1일 때만 실제 Ollama 통합 테스트를 실행합니다.",
    )
    def test_translate_file_chunks_translates_and_saves_markdown(self) -> None:
        """실제 Ollama로 Markdown을 청킹 번역하고 결과 파일을 저장한다."""
        tool = TranslateTool(
            progress=print,
            output_directory=TRANSLATION_OUTPUT_DIRECTORY,
        )

        output_path = tool.translate_file(TEST_MARKDOWN_PATH)

        self.assertTrue(output_path.is_file())
        self.assertEqual(output_path.parent, TRANSLATION_OUTPUT_DIRECTORY)
        self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

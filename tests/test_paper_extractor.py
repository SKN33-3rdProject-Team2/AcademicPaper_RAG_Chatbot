"""PaperExtractor 단위 테스트 (PDF와 네트워크 없이 도는 부분만)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.feature.paper_extractor import (
    ExtractionResult,
    PaperExtractionError,
    PaperExtractor,
    extract_paper_text,
)


def make_span(text, size, y0, y1):
    """PyMuPDF 가 돌려주는 span 모양을 흉내낸다."""
    return {"text": text, "size": size, "bbox": (0.0, y0, 10.0, y1)}


class SafeTitleTest(unittest.TestCase):
    def test_matches_download_filename_rule(self):
        """search_list.py 가 PDF를 저장할 때 쓰는 규칙과 같아야 한다."""
        title = "Towards Goal-oriented Prompt Engineering for Large Language Models: A Survey"
        self.assertEqual(len(PaperExtractor.safe_title(title)), 60)
        self.assertNotIn(":", PaperExtractor.safe_title(title))

    def test_short_title_is_untouched(self):
        self.assertEqual(
            PaperExtractor.safe_title("Six Challenges for Neural Machine Translation"),
            "Six Challenges for Neural Machine Translation",
        )


class SubscriptRestorationTest(unittest.TestCase):
    """B: 글자 크기와 세로 위치로 위·아래 첨자를 되살리는 부분."""

    def test_superscript_is_restored(self):
        spans = [make_span("x", 10.0, 0.0, 10.0), make_span("2", 7.0, 0.0, 6.0)]
        self.assertEqual(PaperExtractor._render_line(spans), "x^{2}")

    def test_subscript_is_restored(self):
        spans = [make_span("W", 10.0, 0.0, 10.0), make_span("i", 7.0, 5.0, 11.0)]
        self.assertEqual(PaperExtractor._render_line(spans), "W_{i}")

    def test_consecutive_superscript_spans_are_merged(self):
        """(z^{(i)}) 가 (z^{(}^{i}^{)}) 로 쪼개지면 안 된다."""
        spans = [
            make_span("z", 10.0, 0.0, 10.0),
            make_span("(", 7.0, 0.0, 6.0),
            make_span("i", 7.0, 0.0, 6.0),
            make_span(")", 7.0, 0.0, 6.0),
        ]
        self.assertEqual(PaperExtractor._render_line(spans), "z^{(i)}")

    def test_consecutive_subscript_spans_are_merged(self):
        spans = [
            make_span("1", 10.0, 0.0, 10.0),
            make_span("[0", 7.0, 5.0, 11.0),
            make_span(",", 7.0, 5.0, 11.0),
            make_span("∞)", 7.0, 5.0, 11.0),
        ]
        self.assertEqual(PaperExtractor._render_line(spans), "1_{[0,∞)}")

    def test_body_text_is_left_alone(self):
        spans = [make_span("The basic concept", 10.0, 0.0, 10.0)]
        self.assertEqual(PaperExtractor._render_line(spans), "The basic concept")

    def test_uniformly_small_line_is_not_marked(self):
        """캡션이나 각주처럼 줄 전체가 작은 경우는 첨자가 아니다."""
        spans = [make_span("Figure 1: caption text", 7.0, 0.0, 7.0)]
        self.assertEqual(PaperExtractor._render_line(spans), "Figure 1: caption text")

    def test_long_run_is_not_treated_as_script(self):
        spans = [
            make_span("body", 10.0, 0.0, 10.0),
            make_span("a very long small fragment", 7.0, 0.0, 6.0),
        ]
        self.assertNotIn("^{", PaperExtractor._render_line(spans))


class RefinementRuleTest(unittest.TestCase):
    def test_ligatures_are_normalized(self):
        self.assertEqual(PaperExtractor._normalize("a ﬁxed oﬀer"), "a fixed offer")

    def test_mapsto_glyph_is_repaired(self):
        self.assertEqual(PaperExtractor._normalize("x 7→ y"), "x ↦ y")

    def test_line_break_hyphen_is_joined(self):
        self.assertEqual(PaperExtractor._join_hyphenation("perfor- mance"), "performance")

    def test_compound_word_keeps_hyphen(self):
        self.assertEqual(PaperExtractor._join_hyphenation("left-to- right"), "left-to-right")

    def test_numbered_heading(self):
        self.assertEqual(PaperExtractor._as_heading("3 Experimental Setup"), "## 3 Experimental Setup")

    def test_known_heading(self):
        self.assertEqual(PaperExtractor._as_heading("Abstract"), "## Abstract")

    def test_street_address_is_not_heading(self):
        self.assertIsNone(PaperExtractor._as_heading("1101 Kitchawan Rd, Yorktown Heights, NY 10598"))

    def test_page_number_and_stamp_are_noise(self):
        self.assertTrue(PaperExtractor._is_noise("7", set()))
        self.assertTrue(PaperExtractor._is_noise("arXiv:1702.01806v2  [cs.CL]  14 Jun 2017", set()))
        self.assertTrue(PaperExtractor._is_noise("Proceedings of ACL", {"Proceedings of ACL"}))

    def test_blank_line_is_not_noise(self):
        self.assertFalse(PaperExtractor._is_noise("   ", set()))

    def test_caption_becomes_quote(self):
        extractor = PaperExtractor(output_dir=Path(tempfile.mkdtemp()))
        page = "Figure 2: German to English results\n\nSome body text follows."
        refined = extractor._refine_page(page, set())
        self.assertIn("> **Figure 2: German to English results**", refined)


class DegenerationGuardTest(unittest.TestCase):
    """비전 판독이 망가졌는지 가려내는 검사. 표를 오탐하면 안 된다."""

    LOCAL = "some local page text " * 30

    def test_unk_run_is_rejected(self):
        text = "Body text. " + "<unk>" * 20
        self.assertIn("<unk>", PaperExtractor._degeneration_reason(text, self.LOCAL))

    def test_empty_response_is_rejected(self):
        self.assertTrue(PaperExtractor._degeneration_reason("   ", self.LOCAL))

    def test_markdown_table_is_not_rejected(self):
        """표 구분행 |:---:|:---:| 이 반복으로 오인되면 표가 통째로 버려진다."""
        table = (
            "| pruning | beam size | speed up | BLEU | TER | fan out | total | sent |\n"
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
            + "".join(
                f"| rp={i / 10} | 5 | {i}% | 27.3 | 54.6 | 4.5 | 122 | 25 |\n"
                for i in range(1, 9)
            )
        )
        self.assertEqual(PaperExtractor._degeneration_reason(table, self.LOCAL), "")

    def test_repeated_sentence_outside_table_is_rejected(self):
        text = "The decoder expands candidates. " * 12
        self.assertTrue(PaperExtractor._degeneration_reason(text, self.LOCAL))

    def test_wildly_inflated_output_is_rejected(self):
        text = "unique words here " * 400
        reason = PaperExtractor._degeneration_reason(text, self.LOCAL)
        self.assertTrue(reason)

    def test_normal_page_passes(self):
        text = "## Introduction\n\nNeural machine translation reached parity with SMT."
        self.assertEqual(PaperExtractor._degeneration_reason(text, self.LOCAL), "")


class ContentPageCountTest(unittest.TestCase):
    """표를 담을지 정하는 기준이 되는, 참고문헌을 뺀 본문 쪽 수."""

    def test_references_heading_marks_the_boundary(self):
        pages = ["body", "body", "References\n\n[1] Someone. 2017.", "[2] Another. 2018."]
        self.assertEqual(PaperExtractor._content_page_count(pages), 2)

    def test_bare_citation_list_marks_the_boundary(self):
        """물리·수학 학술지는 References 제목 없이 [1] 부터 바로 나열한다."""
        citations = "\n".join(f"[{i}] Author {i}, Journal {i} (2020)." for i in range(1, 9))
        pages = ["body", "body", "body", citations]
        self.assertEqual(PaperExtractor._content_page_count(pages), 3)

    def test_paper_without_references_counts_every_page(self):
        self.assertEqual(PaperExtractor._content_page_count(["a", "b", "c"]), 3)

    def test_scattered_citations_do_not_trigger(self):
        """본문 중간의 인용 한둘로 참고문헌 시작을 오판하면 안 된다."""
        pages = ["text [1] and [2] here", "more text", "References", "[1] x"]
        self.assertEqual(PaperExtractor._content_page_count(pages), 2)


class StorageTest(unittest.TestCase):
    """경로를 주입할 수 있으므로 실제 데이터 없이 저장 동작을 확인한다."""

    def _extractor(self, directory):
        return PaperExtractor(
            pdf_dir=Path(directory) / "pdf",
            metadata_db=Path(directory) / "meta.db",
            output_dir=Path(directory) / "out",
        )

    def test_save_writes_db_and_title_json(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor = self._extractor(directory)
            extractor._save(
                ExtractionResult(
                    id="1234v1",
                    title="어떤 논문",
                    source_pdf="paper.pdf",
                    content="## Abstract\n\n본문",
                    n_pages=3,
                )
            )

            self.assertTrue(extractor.is_extracted("1234v1"))
            record = extractor.get("1234v1")
            self.assertEqual(record["title"], "어떤 논문")
            self.assertEqual(record["content"], "## Abstract\n\n본문")
            self.assertEqual(record["n_pages"], 3)
            self.assertEqual(extractor.titles(), {"1234v1": "어떤 논문"})

    def test_second_save_replaces_first(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor = self._extractor(directory)
            for content in ("처음", "나중"):
                extractor._save(
                    ExtractionResult(
                        id="1234v1", title="제목", source_pdf="p.pdf", content=content, n_pages=1
                    )
                )
            self.assertEqual(extractor.get("1234v1")["content"], "나중")
            # sqlite3.connect 의 컨텍스트 매니저는 커밋만 하고 닫지 않는다.
            with closing(sqlite3.connect(extractor.db_path)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM extracted").fetchone()[0], 1)

    def test_markdown_file_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor = self._extractor(directory)
            extractor._save(
                ExtractionResult(
                    id="1234v1",
                    title="어떤 논문",
                    source_pdf="paper.pdf",
                    content="## Abstract\n\n본문",
                    n_pages=3,
                )
            )
            written = extractor.markdown_path("1234v1").read_text(encoding="utf-8")
            self.assertTrue(written.startswith("# 어떤 논문"))
            self.assertIn("- **arXiv ID**: 1234v1", written)
            self.assertIn("- **쪽 수**: 3", written)
            self.assertTrue(written.rstrip().endswith("## Abstract\n\n본문"))

    def test_json_holds_saved_pdf_titles(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor = self._extractor(directory)
            extractor._save(
                ExtractionResult(
                    id="1234v1", title="제목", source_pdf="paper.pdf", content="본문", n_pages=1
                )
            )
            payload = json.loads(extractor.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["1234v1"]["title"], "제목")
            self.assertEqual(payload["1234v1"]["source_pdf"], "paper.pdf")

    def test_missing_metadata_db_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor = self._extractor(directory)
            with self.assertRaises(PaperExtractionError):
                extractor.list_papers()


class ToolInterfaceTest(unittest.TestCase):
    """LangChain 에이전트에 그대로 넘길 수 있어야 한다."""

    def test_tool_is_exposed_with_schema(self):
        self.assertEqual(extract_paper_text.name, "extract_paper_text")
        self.assertIn("paper_ids", extract_paper_text.args)
        self.assertTrue(extract_paper_text.description.strip())


if __name__ == "__main__":
    unittest.main()

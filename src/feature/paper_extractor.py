"""논문 PDF 본문 추출 도구.

data/paper_save 의 PDF 중 사용자가 고른 논문을 팀 정제 규칙에 따라 가공하고,
그 결과를 data/paper_extract 의 DB와 JSON에 저장한다.

쪽을 이미지로 만들어 비전 모델에 넣고, 모든 쪽에 같은 지시문을 적용한다. 그래서
어떤 논문을 넣어도 결과 형태가 일정하고, 분수·근호처럼 평문 추출로는 살릴 수 없는
표기까지 LaTeX 으로 복원된다.

모델을 부르지 못하는 쪽은 로컬 PyMuPDF 추출로 되돌린다. 키가 없거나 서버가 죽어도
파이프라인 전체가 멈추지는 않는다. ``use_vision=False`` 로 로컬 경로만 쓸 수도 있다.

다른 파일에서 그대로 가져다 쓸 수 있다::

    from feature.paper_extractor import PaperExtractor, extract_paper_text

    extractor = PaperExtractor()
    for ref in extractor.list_papers():
        print(ref.id, ref.title)
    result = extractor.extract("1702.01806v1")

LangChain 에이전트에는 ``extract_paper_text`` 를 도구로 넘기면 된다.
"""

from __future__ import annotations

import base64
import json
import re
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------
# 🚨 [모듈 임포트 경로 설정: src 와 프로젝트 루트를 기준으로 맞춤]
# 현재 위치: src/feature/paper_extractor.py
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
for _path in (SRC_DIR, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from langchain_core.tools import tool

from log.app_logger import AppLogger
from log.log_codes import LogCode
from services.nvidia_service import (
    NVIDIA_CONFIG,
    NvidiaServiceError,
    describe_image,
    is_available,
)
# ---------------------------------------------------------------------

logger = AppLogger(__name__)

DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "paper_save"
DEFAULT_METADATA_DB = PROJECT_ROOT / "data" / "paper_list" / "saved_papers.db"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "paper_extract"

# ── 팀 정제 규칙에 쓰이는 표들 ──────────────────────────────────────────

# PDF 조판에 쓰인 합자와 특수 공백. 그대로 두면 번역과 검색에서 모두 깨진다.
TEXT_REPLACEMENTS = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
    " ": " ", " ": " ", " ": " ", "­": "",
    # LaTeX 의 \mapsto 는 조합 글리프라 평문으로 뽑으면 "7→" 로 읽힌다.
    "7→": "↦",
}

CAPTION_PATTERN = re.compile(
    r"^\s*(fig(?:ure)?\.?|table|algorithm|listing)\s*\.?\s*\d+", re.IGNORECASE
)
ARXIV_STAMP_PATTERN = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}v?\d*\s*\[[^\]]+\]", re.IGNORECASE)
REFERENCE_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$", re.IGNORECASE
)
# 섹션 번호는 한두 자리다. 자릿수를 열어두면 "1101 Kitchawan Rd" 같은 주소가 제목으로 잡힌다.
NUMBERED_HEADING_PATTERN = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Z][^.]{2,60})\s*$")
UPPER_HEADING_PATTERN = re.compile(r"^\s*([A-Z][A-Z \-]{3,40})\s*$")
KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work", "method", "methods",
    "methodology", "approach", "experiments", "experimental setup", "results",
    "evaluation", "discussion", "conclusion", "conclusions", "limitations",
    "acknowledgments", "acknowledgements", "appendix",
}

# 위·아래 첨자 판정 기준. 본문보다 이 비율 이하로 작고, 세로로 치우쳐 있어야 한다.
SUBSCRIPT_SIZE_RATIO = 0.85
SUBSCRIPT_OFFSET_RATIO = 0.12
SUBSCRIPT_MAX_CHARS = 12
# 적분·합 기호처럼 드물면서 유독 큰 글리프를 본문 크기 후보에서 걸러내는 기준.
OVERSIZED_GLYPH_SHARE = 0.05
OVERSIZED_GLYPH_RATIO = 1.4
# 표 탐지가 빈 격자를 물어오는 일이 있어, 이 비율 이상 칸이 차 있어야 표로 인정한다.
TABLE_MIN_FILLED_RATIO = 0.4

# 쪽은 서로 독립이라 동시에 보낼 수 있다. 16쪽 논문으로 재보면 워커 4개는 349초,
# 8개는 67초, 16개는 132초였다. 8개를 넘기면 서버가 포화되어 오히려 느려진다.
DEFAULT_VISION_WORKERS = 8

# 비전 판독이 반복 루프에 빠졌는지 가려내는 신호들. 호출이 성공해도 내용이 망가질 수 있다.
UNK_RUN_PATTERN = re.compile(r"(?:<unk>\s*){4,}")
REPEATED_RUN_PATTERN = re.compile(r"(.{2,40}?)\1{6,}", re.DOTALL)
TABLE_ROW_PATTERN = re.compile(r"^[ \t]*\|.*$", re.MULTILINE)
# 마크다운 문법이 붙어 로컬 추출보다 길어지는 건 정상이지만, 몇 배가 되면 반복이다.
VISION_LENGTH_LIMIT = 3.0

# 참고문헌을 뺀 본문이 이 쪽 수 이하일 때만 표를 담는다. 긴 논문은 표·그림을 모두 버려
# 본문만 남긴다. 긴 논문에서 표까지 옮기면 분량과 처리 시간이 감당이 안 되기 때문이다.
MAX_PAGES_FOR_TABLES = 8
REFERENCE_PAGE_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s*)?(?:references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE
)
# 물리·수학 학술지 양식은 "References" 제목 없이 [1] 부터 바로 나열하는 일이 흔하다.
# 제목만 찾으면 참고문헌이 통째로 본문으로 세어진다.
CITATION_ENTRY_PATTERN = re.compile(r"^\s*\[\d{1,3}\]\s", re.MULTILINE)
CITATION_ENTRIES_PER_PAGE = 5

# 어떤 논문을 넣어도 같은 모양으로 나오게 하는 지시문. 일관성의 핵심 손잡이다.
VISION_PROMPT = (
    "Transcribe this page of an academic paper as GitHub-flavored Markdown.\n"
    "Rules:\n"
    "1. Write every formula as LaTeX: inline math as $...$, display math as $$...$$.\n"
    "   Use \\frac, \\sqrt, \\sum, \\mathbb, ^{} and _{} exactly as the page shows.\n"
    "2. Render tables as Markdown tables with a header row. Never use LaTeX tabular.\n"
    "3. Section titles become Markdown headings (## for sections, ### for subsections).\n"
    "4. Keep figure and table captions as normal lines starting with 'Figure N:' or 'Table N:'.\n"
    "   Do not describe the figure image itself.\n"
    "5. Drop page numbers, running headers, footers and the arXiv stamp.\n"
    "6. Transcribe every word. Never summarize, translate, or add commentary.\n"
    "7. Output only the Markdown, with no preamble and no surrounding code fence."
)

# 긴 논문에서 쓰는 지시문. 본문 서술만 남기고 표와 그림은 통째로 건너뛴다.
VISION_PROMPT_TEXT_ONLY = (
    "Transcribe this page of an academic paper as GitHub-flavored Markdown.\n"
    "Rules:\n"
    "1. Write every formula as LaTeX: inline math as $...$, display math as $$...$$.\n"
    "   Use \\frac, \\sqrt, \\sum, \\mathbb, ^{} and _{} exactly as the page shows.\n"
    "2. Skip tables and figures entirely. Do not transcribe table contents, do not write\n"
    "   Markdown tables, and do not write figure or table captions.\n"
    "3. Section titles become Markdown headings (## for sections, ### for subsections).\n"
    "4. Drop page numbers, running headers, footers and the arXiv stamp.\n"
    "5. Transcribe every sentence of the running text. Never summarize or add commentary.\n"
    "6. Output only the Markdown, with no preamble and no surrounding code fence."
)


class PaperExtractionError(RuntimeError):
    """논문 본문 추출을 완료하지 못했을 때 발생한다."""


@dataclass(frozen=True)
class PaperRef:
    """추출 대상 논문 하나."""

    id: str
    title: str
    pdf_path: Path


@dataclass(frozen=True)
class ExtractionResult:
    """추출·가공 결과."""

    id: str
    title: str
    source_pdf: str
    content: str
    n_pages: int
    n_vision_pages: int = 0
    skipped: bool = False

    @property
    def n_chars(self) -> int:
        return len(self.content)

    @property
    def extractor(self) -> str:
        """이 결과가 어떤 방식으로 나왔는지. 나중에 행만 보고도 구분할 수 있어야 한다."""
        if not self.n_vision_pages:
            return "pymupdf"
        if self.n_vision_pages == self.n_pages:
            return "vision"
        return f"mixed({self.n_vision_pages}/{self.n_pages})"


class PaperExtractor:
    """PDF를 팀 규칙대로 가공해 저장하는 도구.

    경로를 생성자로 받으므로 테스트나 다른 프로젝트에서도 그대로 쓸 수 있다.
    무거운 의존성(PyMuPDF)은 실제로 추출할 때만 불러온다.
    """

    def __init__(
        self,
        *,
        pdf_dir: str | Path | None = None,
        metadata_db: str | Path | None = None,
        output_dir: str | Path | None = None,
        use_vision: bool = True,
        vision_workers: int = DEFAULT_VISION_WORKERS,
    ) -> None:
        self.pdf_dir = Path(pdf_dir) if pdf_dir else DEFAULT_PDF_DIR
        self.metadata_db = Path(metadata_db) if metadata_db else DEFAULT_METADATA_DB
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.db_path = self.output_dir / "extracted_papers.db"
        self.json_path = self.output_dir / "extracted_papers.json"
        self.use_vision = use_vision
        self.vision_workers = max(1, vision_workers)
        # DB에 제목이 없는 PDF는 파일 안에 든 제목을 대신 쓴다.
        self._embedded_title = ""

    # ── 조회 ────────────────────────────────────────────────────────

    @staticmethod
    def safe_title(title: str) -> str:
        """search_list.py 가 PDF를 저장할 때 쓰는 파일명 규칙과 같게 정규화한다.

        다운로드된 PDF에는 arXiv id가 없어, 제목을 같은 규칙으로 정규화하는 것이
        PDF와 논문 메타데이터를 잇는 유일한 결정적 방법이다.
        """
        cleaned = "".join(c for c in title if c.isalnum() or c in " _-").rstrip()
        return cleaned[:60]

    def list_papers(self) -> list[PaperRef]:
        """PDF가 실제로 존재하는 논문만 돌려준다."""
        if not self.metadata_db.is_file():
            raise PaperExtractionError(
                f"논문 메타데이터 DB가 없습니다: {self.metadata_db}\n"
                "먼저 src/feature/search.py 로 논문을 검색·저장해 주세요."
            )

        with closing(sqlite3.connect(self.metadata_db)) as conn:
            rows = conn.execute("SELECT id, title FROM papers").fetchall()

        by_path: dict[Path, PaperRef] = {}
        for paper_id, title in rows:
            path = self.pdf_dir / f"{self.safe_title(title or '')}.pdf"
            if path.is_file():
                by_path[path] = PaperRef(id=paper_id, title=title, pdf_path=path)

        # 사용자가 직접 넣어둔 PDF도 잡는다. Agent 1 을 거치지 않고 파일만 떨어뜨리는
        # 경우가 있고, 그때 파일명은 보통 arXiv id 다.
        if self.pdf_dir.is_dir():
            for path in sorted(self.pdf_dir.glob("*.pdf")):
                by_path.setdefault(
                    path, PaperRef(id=path.stem, title=path.stem, pdf_path=path)
                )

        refs = list(by_path.values())
        refs.sort(key=lambda ref: ref.title.lower())
        return refs

    def find(self, paper_id: str) -> PaperRef:
        """arXiv id 로 논문 하나를 찾는다."""
        for ref in self.list_papers():
            if ref.id == paper_id:
                return ref
        raise PaperExtractionError(f"'{paper_id}' 에 해당하는 PDF가 data/paper_save 에 없습니다.")

    def is_extracted(self, paper_id: str) -> bool:
        if not self.db_path.is_file():
            return False
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM extracted WHERE id = ?", (paper_id,)
            ).fetchone()
        return row is not None

    def get(self, paper_id: str) -> dict | None:
        """저장된 가공본을 읽는다. 다음 단계(번역·요약·색인)의 진입점이다."""
        if not self.db_path.is_file():
            return None
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM extracted WHERE id = ?", (paper_id,)
            ).fetchone()
        return dict(row) if row else None

    def titles(self) -> dict[str, str]:
        """저장한 논문의 id → 제목."""
        if not self.json_path.is_file():
            return {}
        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {key: value.get("title", "") for key, value in data.items()}

    # ── 추출 ────────────────────────────────────────────────────────

    def extract(self, paper_id: str, *, force: bool = False) -> ExtractionResult:
        """논문 하나를 가공해 저장한다."""
        if self.is_extracted(paper_id) and not force:
            record = self.get(paper_id) or {}
            logger.log(LogCode.PAPER_EXTRACTION_SKIPPED, paper_id=paper_id)
            return ExtractionResult(
                id=paper_id,
                title=record.get("title", ""),
                source_pdf=record.get("source_pdf", ""),
                content=record.get("content", ""),
                n_pages=int(record.get("n_pages") or 0),
                n_vision_pages=int(record.get("n_pages") or 0)
                if str(record.get("extractor", "")).startswith("vision")
                else 0,
                skipped=True,
            )

        try:
            ref = self.find(paper_id)
        except PaperExtractionError:
            logger.log(LogCode.PAPER_EXTRACTION_REJECTED, paper_id=paper_id)
            raise

        logger.log(LogCode.PAPER_EXTRACTION_STARTED, paper_id=paper_id, title=ref.title)
        try:
            pages, n_vision, with_tables = self._read_pdf(ref.pdf_path)
            content = self._refine(pages, with_tables)
        except PaperExtractionError:
            raise
        except Exception as exc:
            logger.log(LogCode.PAPER_EXTRACTION_FAILED, paper_id=paper_id, reason=str(exc))
            raise PaperExtractionError(f"'{paper_id}' 추출에 실패했습니다: {exc}") from exc

        # DB를 거치지 않은 PDF는 제목이 파일명이다. 파일에 든 제목이 있으면 그쪽이 낫다.
        title = ref.title
        if title == ref.pdf_path.stem and self._embedded_title:
            title = self._embedded_title

        result = ExtractionResult(
            id=ref.id,
            title=title,
            source_pdf=ref.pdf_path.name,
            content=content,
            n_pages=len(pages),
            n_vision_pages=n_vision,
        )
        self._save(result)
        logger.log(
            LogCode.PAPER_EXTRACTION_SUCCEEDED,
            paper_id=paper_id,
            pages=result.n_pages,
            chars=result.n_chars,
            extractor=result.extractor,
        )
        return result

    def extract_many(self, paper_ids: list[str], *, force: bool = False) -> list[ExtractionResult]:
        """여러 논문을 차례로 가공한다. 하나가 실패해도 나머지는 계속한다."""
        results: list[ExtractionResult] = []
        for paper_id in paper_ids:
            try:
                results.append(self.extract(paper_id, force=force))
            except PaperExtractionError:
                continue
        return results

    # ── PDF 읽기 ────────────────────────────────────────────────────

    @staticmethod
    def _import_pymupdf():
        """PyMuPDF를 실제로 필요할 때만 불러온다. 임포트만으로는 요구하지 않는다."""
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf  # PyMuPDF 1.24 이전의 옛 모듈명
            except ImportError as exc:
                raise PaperExtractionError(
                    "PyMuPDF가 설치돼 있지 않습니다. "
                    "`pip install -r requirements.txt` 를 먼저 실행해 주세요."
                ) from exc
        return pymupdf

    @staticmethod
    def _table_is_useful(table) -> bool:
        """표 탐지는 빈 격자를 잡아내기도 한다. 알맹이가 있는 것만 쓴다."""
        try:
            rows = table.extract()
        except Exception:
            return False
        if len(rows) < 2 or table.col_count < 2:
            return False
        cells = [cell for row in rows for cell in row]
        if not cells:
            return False
        filled = sum(1 for cell in cells if cell and str(cell).strip())
        return filled / len(cells) >= TABLE_MIN_FILLED_RATIO

    def _read_pdf(self, pdf_path: Path) -> tuple[list[str], int, bool]:
        """페이지별 텍스트를 뽑는다.

        표는 PyMuPDF 의 표 탐지로 마크다운 표로 살리고, 나머지 본문은 span 단위로
        읽어 위·아래 첨자를 되살린다. 그림은 버린다.

        (쪽별 텍스트, 비전으로 읽어낸 쪽 수, 표 포함 여부) 를 돌려준다.
        참고문헌을 뺀 본문이 짧은 논문에서만 표를 담고, 긴 논문은 표·그림을 버린다.

        표 탐지와 첨자 복원을 함께 쓰는 이유는 어느 한쪽만으로는 부족해서다.
        pymupdf4llm 은 표를 잘 뽑지만 아래첨자를 통째로 잃고(``∇_θΦ_a`` 가
        ``_∇θ_ Φ _a_`` 가 된다), span 단위 추출만으로는 표가 세로로 흩어진다.
        """
        pymupdf = self._import_pymupdf()
        if not pdf_path.is_file():
            raise PaperExtractionError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        with pymupdf.open(pdf_path) as document:
            self._embedded_title = (document.metadata or {}).get("title", "").strip()
            local_pages = [self._read_page(page) for page in document]
            content_pages = self._content_page_count(local_pages)
            with_tables = content_pages <= MAX_PAGES_FOR_TABLES

            if not self.use_vision or not is_available():
                if not is_available():
                    logger.log(LogCode.PAGE_VISION_UNAVAILABLE)
                return [self._normalize(page) for page in local_pages], 0, with_tables

            dpi = int(NVIDIA_CONFIG.get("render_dpi", 150))
            images = [
                base64.b64encode(page.get_pixmap(dpi=dpi).tobytes("png")).decode()
                for page in document
            ]

        pages, n_vision = self._read_pages_with_vision(images, local_pages, with_tables)
        return [self._normalize(page) for page in pages], n_vision, with_tables

    @staticmethod
    def _content_page_count(local_pages: list[str]) -> int:
        """참고문헌을 뺀 본문 쪽 수.

        참고문헌이 시작되는 쪽부터는 본문으로 세지 않는다. 논문 뒤쪽의 참고문헌은
        몇 쪽씩 차지하면서도 표·그림 판단과는 무관하기 때문이다.

        "References" 제목을 찾거나, 제목 없이 ``[1]`` 부터 나열하는 양식이면
        인용 항목이 빽빽한 쪽을 경계로 본다.
        """
        for index, page in enumerate(local_pages):
            if REFERENCE_PAGE_PATTERN.search(page):
                return index
            if len(CITATION_ENTRY_PATTERN.findall(page)) >= CITATION_ENTRIES_PER_PAGE:
                return index
        return len(local_pages)

    @staticmethod
    def _degeneration_reason(text: str, local_text: str) -> str:
        """비전 판독이 망가졌으면 그 이유를, 멀쩡하면 빈 문자열을 돌려준다.

        호출이 200으로 성공해도 모델이 반복 루프에 빠질 수 있다. 실제로 참고문헌
        쪽에서 ``<unk>`` 를 만 자 넘게 뱉어 분량이 두 배가 된 적이 있다.
        """
        if not text.strip():
            return "빈 응답"
        if UNK_RUN_PATTERN.search(text):
            return "<unk> 반복"
        # 반복 검사에서는 표를 빼고 본다. 마크다운 표의 |:---:|:---:| 구분행과 규칙적인
        # 칸 배열이 반복으로 오인돼, 표가 잘 나온 쪽이 오히려 버려진다.
        if REPEATED_RUN_PATTERN.search(TABLE_ROW_PATTERN.sub("", text)):
            return "같은 조각 반복"
        local_length = len(local_text.strip())
        if local_length > 200 and len(text) > local_length * VISION_LENGTH_LIMIT:
            return f"분량 과다 ({len(text)}자 vs 로컬 {local_length}자)"
        return ""

    def _read_pages_with_vision(
        self, images: list[str], local_pages: list[str], with_tables: bool
    ) -> tuple[list[str], int]:
        """쪽 이미지를 모델에 보내 마크다운으로 받는다.

        같은 지시문을 모든 쪽에 똑같이 적용하므로 어떤 논문을 넣어도 결과 형태가
        일정하다. 쪽은 서로 독립이라 몇 개씩 동시에 보내 시간을 줄이고,
        실패한 쪽만 로컬 추출 결과로 되돌린다.

        (쪽별 결과, 비전으로 읽어낸 쪽 수) 를 돌려준다. 쪽 수를 함께 남기는 이유는,
        나중에 저장된 행만 보고도 어떤 방식으로 뽑았는지 알 수 있어야 하기 때문이다.
        """

        prompt = VISION_PROMPT if with_tables else VISION_PROMPT_TEXT_ONLY

        def read_one(index: int) -> tuple[str, bool]:
            local = local_pages[index]
            try:
                # 폴백된 쪽은 수식·표 품질이 떨어지므로, 동시요청 제한(16/16)에 걸린 경우
                # 잠시 기다렸다 더 끈질기게 다시 시도한다.
                text = describe_image(
                    images[index], prompt, max_tokens=4096, retries=4
                ).strip()
            except NvidiaServiceError as exc:
                logger.log(LogCode.PAGE_VISION_FALLBACK, page=index + 1, reason=str(exc)[:120])
                return local, False

            flaw = self._degeneration_reason(text, local)
            if flaw:
                logger.log(LogCode.PAGE_VISION_FALLBACK, page=index + 1, reason=flaw)
                return local, False
            return text, True

        with ThreadPoolExecutor(max_workers=self.vision_workers) as pool:
            outcomes = list(pool.map(read_one, range(len(images))))
        return [text for text, _ in outcomes], sum(1 for _, ok in outcomes if ok)

    @staticmethod
    def _render_table(table) -> str:
        """표를 마크다운으로 옮긴다.

        칸 내용은 PyMuPDF 의 ``to_markdown()`` 을 그대로 쓴다. ``extract()`` 로 직접
        조립해 보면 칸 안의 순서가 흐트러져 ``1(0,∞)(x)`` 가 ``1 (x) (0,∞)`` 가 된다.

        이탤릭 ``_..._`` 는 지운다. 본문에서 아래첨자로 쓰는 ``_{...}`` 와 생김새가
        겹쳐, 그대로 두면 ``max_{_0_, x}_`` 처럼 엉키기 때문이다.

        남는 한계: 칸 안에 ``|`` 가 있으면(예: 절댓값 ``1+|x|``) 그 줄만 열 수가
        어긋난다. 글자는 온전하고 그 줄에만 생기는 문제라 순서 보존을 택했다.
        """
        markdown = table.to_markdown().strip()
        return markdown.replace("~~", "").replace("_", "")

    def _read_page(self, page) -> str:
        """한 쪽을 표와 본문으로 나눠 읽고, 원래 세로 순서대로 이어 붙인다."""
        table_areas: list[tuple[float, float, float, str]] = []
        try:
            for table in page.find_tables().tables:
                if not self._table_is_useful(table):
                    continue
                bbox = table.bbox
                rendered = self._render_table(table)
                if rendered:
                    table_areas.append(
                        (float(bbox[1]), float(bbox[3]), float(bbox[0]), rendered)
                    )
        except Exception:
            table_areas = []  # 표 탐지는 보조 수단이라, 실패해도 본문은 계속 읽는다

        pieces: list[tuple[float, str]] = [(top, markdown) for top, _, _, markdown in table_areas]

        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0 = 텍스트, 1 = 이미지 → 그림은 버린다
                continue
            block_top = float(block["bbox"][1])
            block_bottom = float(block["bbox"][3])
            # 표 안에 있는 글자는 이미 표로 옮겼으므로 본문에서 뺀다.
            if any(top <= block_top and block_bottom <= bottom for top, bottom, _, _ in table_areas):
                continue

            lines = [
                rendered
                for line in block.get("lines", [])
                if (rendered := self._render_line(line.get("spans", []))).strip()
            ]
            if lines:
                pieces.append((block_top, "\n".join(lines)))

        pieces.sort(key=lambda item: item[0])
        return "\n".join(text for _, text in pieces)

    @staticmethod
    def _render_line(spans: list[dict]) -> str:
        """한 줄을 글자로 옮기면서 위·아래 첨자를 ^{} / _{} 로 복원한다.

        PyMuPDF 의 평문 추출은 첨자를 본문과 같은 높이로 흘려보내 ``W^{(L)}`` 이
        ``W(L)`` 이 된다. span 의 글자 크기와 세로 위치가 남아 있으므로, 본문보다
        작고 위나 아래로 치우친 조각을 첨자로 판정한다. 모델 없이 결정적으로 된다.
        """
        weights: Counter[float] = Counter()
        for span in spans:
            text = span.get("text", "")
            if text.strip():
                weights[round(float(span.get("size", 0)), 1)] += len(text.strip())
        if not weights:
            return ""

        # 본문 크기는 원칙적으로 그 줄의 가장 큰 글자 크기다. 첨자는 정의상 본문보다 작다.
        # 다만 적분·합 기호처럼 혼자만 훌쩍 큰 글리프가 있으면 본문이 통째로 첨자로 몰리므로,
        # 아주 드물면서 유독 큰 크기는 후보에서 걸러낸다.
        total = sum(weights.values())
        ordered = sorted(weights, reverse=True)
        body_size = ordered[0]
        for index, size in enumerate(ordered):
            smaller = ordered[index + 1] if index + 1 < len(ordered) else None
            is_outlier = (
                weights[size] / total < OVERSIZED_GLYPH_SHARE
                and smaller is not None
                and size > smaller * OVERSIZED_GLYPH_RATIO
            )
            if not is_outlier:
                body_size = size
                break
        body_spans = [
            span for span in spans
            if round(float(span.get("size", 0)), 1) == body_size and span.get("text", "").strip()
        ]
        if not body_spans:
            return "".join(span.get("text", "") for span in spans)

        centers = [
            (float(span["bbox"][1]) + float(span["bbox"][3])) / 2 for span in body_spans
        ]
        body_center = sum(centers) / len(centers)
        threshold = body_size * SUBSCRIPT_OFFSET_RATIO

        # 먼저 span 마다 본문/윗첨자/아래첨자를 판정한다.
        marked: list[tuple[str, str]] = []
        for span in spans:
            text = span.get("text", "")
            stripped = text.strip()
            size = float(span.get("size", 0))
            if not stripped or size >= body_size * SUBSCRIPT_SIZE_RATIO:
                marked.append(("body", text))
                continue

            center = (float(span["bbox"][1]) + float(span["bbox"][3])) / 2
            if center < body_center - threshold:
                marked.append(("sup", stripped))
            elif center > body_center + threshold:
                marked.append(("sub", stripped))
            else:
                marked.append(("body", text))

        # 이어진 첨자는 한 덩어리로 묶는다. span 단위로 감싸면 (z^{(i)})^{m}_{i=1} 이
        # (z^{(}^{i}^{)})^{m} _{i}_{=1} 처럼 글자마다 쪼개진다.
        parts: list[str] = []
        index = 0
        while index < len(marked):
            kind, text = marked[index]
            if kind == "body":
                parts.append(text)
                index += 1
                continue

            run = [text]
            index += 1
            while index < len(marked) and marked[index][0] == kind:
                run.append(marked[index][1])
                index += 1

            merged = "".join(run)
            if len(merged) > SUBSCRIPT_MAX_CHARS:  # 너무 길면 첨자가 아니라고 본다
                parts.append(merged)
            else:
                parts.append(f"^{{{merged}}}" if kind == "sup" else f"_{{{merged}}}")
        return "".join(parts)

    # ── 팀 정제 규칙 ────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """조판용 합자와 특수 공백을 일반 문자로 되돌린다."""
        for source, target in TEXT_REPLACEMENTS.items():
            text = text.replace(source, target)
        return text

    @staticmethod
    def _running_lines(pages: list[str]) -> set[str]:
        """여러 쪽에 반복 등장하는 머리말·꼬리말을 찾는다."""
        if len(pages) < 4:
            return set()
        counter: Counter[str] = Counter()
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            for line in lines[:2] + lines[-2:]:
                if 3 <= len(line) <= 90:
                    counter[line] += 1
        threshold = max(3, int(len(pages) * 0.6))
        return {line for line, count in counter.items() if count >= threshold}

    @staticmethod
    def _is_noise(line: str, running: set[str]) -> bool:
        stripped = line.strip()
        if not stripped:
            return False  # 빈 줄은 잡음이 아니라 문단 경계다
        if stripped in running:
            return True
        if stripped.isdigit() and len(stripped) <= 4:  # 쪽번호
            return True
        return bool(ARXIV_STAMP_PATTERN.search(stripped))

    @staticmethod
    def _as_heading(line: str) -> str | None:
        """절 제목으로 보이는 줄을 마크다운 헤딩으로 바꾼다."""
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            return None
        if REFERENCE_HEADING_PATTERN.match(stripped):
            return "## References"

        numbered = NUMBERED_HEADING_PATTERN.match(stripped)
        if numbered:
            depth = numbered.group(1).count(".") + 2
            return f"{'#' * min(depth, 6)} {numbered.group(1)} {numbered.group(2).strip()}"

        if stripped.lower().strip(". ") in KNOWN_HEADINGS:
            return f"## {stripped.strip('. ')}"

        upper = UPPER_HEADING_PATTERN.match(stripped)
        if upper and upper.group(1).strip().lower() in KNOWN_HEADINGS:
            return f"## {upper.group(1).strip().title()}"
        return None

    @staticmethod
    def _join_hyphenation(text: str) -> str:
        """줄바꿈 때문에 끊긴 단어를 붙인다.

        'perfor- mance' 는 하이픈을 지우고 붙여야 하지만, 'left-to- right' 처럼 원래
        합성어였던 것은 하이픈을 살려야 한다. 왼쪽 조각에 이미 하이픈이 있거나
        오른쪽이 대문자로 시작하면 원래 하이픈으로 본다.
        """

        def replace(match: re.Match[str]) -> str:
            left, right = match.group(1), match.group(2)
            if "-" in left or not right[:1].islower():
                return f"{left}-{right}"
            return f"{left}{right}"

        return re.sub(r"([A-Za-z][A-Za-z-]*)-\s+([A-Za-z]\w*)", replace, text)

    def _refine_page(self, page: str, running: set[str], with_tables: bool = True) -> str:
        """한 쪽을 읽을 만한 마크다운으로 정리한다.

        ``with_tables`` 가 False 면 표와 그림 캡션을 모두 버리고 본문만 남긴다.
        비전 지시문에서도 걸러내지만, 로컬 추출로 되돌아온 쪽이 섞일 수 있어
        여기서 한 번 더 거른다.
        """
        refined: list[str] = []
        paragraph_lines: list[str] = []
        table_rows: list[str] = []

        def flush_paragraph() -> None:
            if not paragraph_lines:
                return
            paragraph = self._join_hyphenation(" ".join(paragraph_lines))
            paragraph = re.sub(r"\s{2,}", " ", paragraph).strip()
            if paragraph:
                refined.append(paragraph)
            paragraph_lines.clear()

        def flush_table() -> None:
            # 표는 줄바꿈 하나로 이어져야 한 덩어리로 읽힌다. 빈 줄이 끼면 깨진다.
            if table_rows:
                refined.append("\n".join(table_rows))
                table_rows.clear()

        def flush_all() -> None:
            flush_paragraph()
            flush_table()

        for line in page.splitlines():
            if self._is_noise(line, running):
                continue
            stripped = line.strip()

            # 마크다운 표는 줄 구조가 곧 의미다. 문단으로 합치면 표가 깨진다.
            if stripped.startswith("|"):
                flush_paragraph()
                if with_tables:
                    table_rows.append(stripped)
                continue
            flush_table()

            if not stripped:
                flush_paragraph()
                continue

            heading = self._as_heading(stripped)
            if heading:
                flush_paragraph()
                refined.append(heading)
                continue

            if CAPTION_PATTERN.match(stripped):
                flush_paragraph()
                # 그림 자체는 버리고 캡션만 인용구로 남겨 맥락을 유지한다.
                if with_tables:
                    refined.append(f"> **{stripped}**")
                continue

            paragraph_lines.append(stripped)

        flush_all()
        return "\n\n".join(refined)

    def _refine(self, pages: list[str], with_tables: bool = True) -> str:
        """모든 쪽에 같은 규칙을 적용해 하나의 마크다운으로 합친다."""
        running = self._running_lines(pages)
        refined = [self._refine_page(page, running, with_tables) for page in pages]
        return "\n\n".join(part for part in refined if part.strip())

    # ── 저장 ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extracted (
                    id         TEXT PRIMARY KEY,
                    title      TEXT,
                    source_pdf TEXT,
                    content    TEXT,
                    n_pages    INTEGER,
                    n_chars    INTEGER,
                    extractor  TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def markdown_path(self, paper_id: str) -> Path:
        """사람이 바로 열어볼 수 있는 마크다운 파일 경로."""
        return self.output_dir / f"{paper_id}.md"

    def _write_markdown(self, result: ExtractionResult) -> Path:
        """가공본을 마크다운 파일로도 떨어뜨린다. 앞머리에 출처를 적어 둔다."""
        path = self.markdown_path(result.id)
        header = (
            f"# {result.title}\n\n"
            f"- **arXiv ID**: {result.id}\n"
            f"- **원본 PDF**: {result.source_pdf}\n"
            f"- **쪽 수**: {result.n_pages}\n\n---\n\n"
        )
        path.write_text(header + result.content, encoding="utf-8")
        return path

    def _save(self, result: ExtractionResult) -> None:
        """가공본을 DB에 넣고, 제목 목록 JSON과 마크다운 파일을 함께 갱신한다."""
        self._init_db()
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO extracted "
                "(id, title, source_pdf, content, n_pages, n_chars, extractor, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (
                    result.id,
                    result.title,
                    result.source_pdf,
                    result.content,
                    result.n_pages,
                    result.n_chars,
                    result.extractor,
                ),
            )
        self._write_markdown(result)
        self._rebuild_json()

    def _rebuild_json(self) -> None:
        """DB를 원본으로 삼아 제목 목록 JSON을 통째로 다시 쓴다."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, title, source_pdf FROM extracted ORDER BY created_at DESC"
            ).fetchall()
        payload = {
            row[0]: {"id": row[0], "title": row[1], "source_pdf": row[2]} for row in rows
        }
        self.json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8"
        )


@tool
def extract_paper_text(paper_ids: list[str]) -> dict:
    """사용자가 고른 논문 PDF의 본문을 팀 정제 규칙대로 가공해 저장한다.

    arXiv id 목록을 받아 각 논문의 PDF에서 본문을 뽑고, 그림과 머리말을 걷어내고
    수식의 위·아래 첨자를 되살려 data/paper_extract 에 저장한 뒤 처리 결과를 돌려준다.
    """
    extractor = PaperExtractor()
    results = extractor.extract_many(paper_ids)
    return {
        "extracted": [
            {
                "id": result.id,
                "title": result.title,
                "n_pages": result.n_pages,
                "n_chars": result.n_chars,
                "skipped": result.skipped,
            }
            for result in results
        ],
        "failed": [pid for pid in paper_ids if pid not in {r.id for r in results}],
    }


if __name__ == "__main__":
    _extractor = PaperExtractor()
    _refs = _extractor.list_papers()
    if not _refs:
        print("[System] 📭 data/paper_save 에 PDF가 없습니다.")
        raise SystemExit(1)

    print(f"\n[논문 추출] data/paper_save 의 논문 {len(_refs)}편")
    print("-" * 70)
    for _index, _ref in enumerate(_refs, start=1):
        _mark = "✅" if _extractor.is_extracted(_ref.id) else "  "
        print(f"{_index:>3}. {_mark} {_ref.id:<14} {_ref.title[:44]}")
    print("-" * 70)

    _answer = input("추출할 논문 번호를 쉼표로 구분해 입력하세요 (엔터: 취소): ").strip()
    if not _answer:
        raise SystemExit(0)

    _chosen = [
        _refs[int(_token) - 1].id
        for _token in _answer.replace(" ", "").split(",")
        if _token.isdigit() and 1 <= int(_token) <= len(_refs)
    ]
    for _result in _extractor.extract_many(_chosen):
        _state = "건너뜀" if _result.skipped else "저장 완료"
        print(f"  {_state}: {_result.id} · {_result.n_pages}쪽 · {_result.n_chars:,}자")

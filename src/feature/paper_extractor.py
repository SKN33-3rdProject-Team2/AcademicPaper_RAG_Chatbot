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
from dataclasses import dataclass, field, replace
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
    chat,
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


# 대단원 번호는 아라비아 숫자 한 단계(1, 2) 또는 로마 숫자(I, II)다.
# "2.1" 처럼 점이 있으면 소단원이다.
ARABIC_SECTION_NUMBER = re.compile(r"^\d+$")
ROMAN_SECTION_NUMBER = re.compile(r"^[IVXLCDM]+$")
# 로마 숫자로 읽히지만 소단원 글자로도 쓰이는 것들. C. Tasks 가 대단원으로 승격되던 원인이다.
AMBIGUOUS_LETTERS = set("IVXLCDM")
# 번호가 없어도 대단원인 절 이름들
MAJOR_SECTION_TITLES = {
    "abstract", "introduction", "background", "related work", "method", "methods",
    "methodology", "approach", "experiments", "experimental setup", "results",
    "results and analysis", "evaluation", "discussion", "conclusion", "conclusions",
    "limitations", "acknowledgments", "acknowledgements", "appendix",
}
# 대단원으로 잡되 저장에서는 빼는 절
EXCLUDED_SECTION_TITLES = {"references", "bibliography"}

# IEEE·Springer 양식은 초록 표제를 본문 첫 줄에 붙여 쓴다.
#   "**_Abstract_**--Recent advances in Large Language Models ..."
# 독립된 줄이 아니라 헤딩으로 안 잡히고, 그러면 abstract 컬럼이 통째로 빈다.
INLINE_ABSTRACT_PATTERN = re.compile(
    r"^[*_\s]*(abstract|index terms|keywords)[*_\s]*[—–\-:]{1,3}\s*", re.IGNORECASE
)
# 폴백된 쪽에서는 절 표제가 문단 한가운데 섞인다.
#   "... dependent on the model variant used. VI. CONCLUSION In this work, we ..."
# 번호 + 온점 + 대문자 표제 다음에 보통 문장이 이어지는 자리를 경계로 본다.
INLINE_SECTION_PATTERN = re.compile(
    r"(?:(?<=\.)|(?<=^))\s*((?:[IVXLCDM]{1,5}|\d{1,2})\.)\s+([A-Z][A-Z][A-Z \-]{2,40}?)"
    r"(?=\s+[A-Z][a-z])"
)

# 논문의 표준 골격(IMRaD). 어떤 논문이 와도 이 컬럼에 맞춰 저장한다.
IMRAD_COLUMNS = (
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "result",
    "conclusion",
)
OTHERS_COLUMN = "others"

# 모델 호출이 실패했을 때 쓰는 대비책. 절 이름에 이 낱말이 있으면 해당 구분으로 본다.
SECTION_KEYWORDS = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "motivation"),
    "related_work": ("related work", "related research", "background", "prior work"),
    "method": ("method", "approach", "model", "architecture", "framework", "algorithm",
               "proposed", "theory", "formulation", "preliminaries"),
    "experiment": ("experiment", "setup", "implementation", "dataset", "data",
                   "simulation", "training", "evaluation protocol"),
    "result": ("result", "analysis", "discussion", "evaluation", "ablation", "finding"),
    "conclusion": ("conclusion", "summary", "future work", "outlook", "concluding"),
}

# 지시문은 영어로 쓴다. 판단 대상(영어 절 제목)과 출력(영어 식별자)이 모두 영어이고,
# 활성 파라미터가 작은 모델일수록 영어 지시를 더 잘 따르기 때문이다.
SECTION_CLASSIFY_SYSTEM_PROMPT = (
    "You classify the structure of academic papers. Given the list of a paper's "
    "top-level sections, decide what role each one plays in the paper.\n"
    "[Categories]\n"
    "- abstract: the paper's abstract\n"
    "- introduction: background, motivation, the problem being solved\n"
    "- related_work: survey of prior research\n"
    "- method: the proposed technique, model, architecture, theory, or definitions\n"
    "- experiment: experimental setup, datasets, implementation details, simulation procedure\n"
    "- result: experimental results, analysis, discussion, ablations\n"
    "- conclusion: conclusions, summary, future work\n"
    "- others: none of the above (acknowledgements, appendix, data availability, "
    "author contributions)\n"
    "[Rules]\n"
    "1. Judge by the role the section plays, not by its name. Papers name sections after "
    "their own subject matter, so the name often differs from the standard one. "
    "For example 'Original Beam Search' is method, and 'Direct Numerical Simulations' "
    "is experiment.\n"
    "2. When the title alone is ambiguous, use the opening sentence given with it.\n"
    "3. Several sections may share one category. Papers often split the method across "
    "multiple sections.\n"
    "4. Do not force a fit. When you cannot tell, answer others.\n"
    "5. Classify every section given. Never omit one.\n"
    "6. Reply with JSON only, in the shape below. No preamble, no explanation, "
    "no code fence. Use each section's heading exactly as it appears in the list.\n"
    '{"section heading": "category", "section heading": "category"}'
)

# 절 번호 표기는 논문마다 다르다. "1 Introduction", "I. INTRODUCTION", "B. Trajectories"
# 를 모두 (번호, 제목) 으로 나눈다. 글자 번호는 마침표를 요구해 "A Comparison of ..."
# 같은 관사 시작 제목을 번호로 오인하지 않게 한다.
SECTION_LABEL_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)*\.?|[IVXLC]{1,5}\.|[A-Z]\.)\s+(\S.*)$"
)


class PaperExtractionError(RuntimeError):
    """논문 본문 추출을 완료하지 못했을 때 발생한다."""


@dataclass(frozen=True)
class PaperRef:
    """추출 대상 논문 하나."""

    id: str
    title: str
    pdf_path: Path
    # 메타데이터 DB에서 온 것인지. False 면 사용자가 직접 넣은 PDF 라 제목이 파일명이다.
    from_metadata: bool = True


@dataclass(frozen=True)
class Section:
    """논문의 대단원 하나. 심층 질의응답이 근거로 가리키는 단위다.

    컬럼을 고정하지 않고 논문이 실제로 가진 대단원을 순서대로 담는다. 논문마다
    절 이름이 달라(``5 Experiments`` vs ``I. INTRODUCTION``) 고정 컬럼으로는
    담을 수 없기 때문이다. 소단원은 자기 대단원 안에 이어 붙이고,
    참고문헌은 담지 않는다.
    """

    no: str                  # 절 번호 ("1", "I"). 없으면 빈 문자열
    title: str               # 절 제목 ("Introduction")
    pages: tuple[int, ...]   # 이 절이 걸쳐 있는 원본 쪽번호
    text: str                # 절 본문 (소단원 포함)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "no": self.no,
            "title": self.title,
            "pages": list(self.pages),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Section":
        return cls(
            no=payload.get("no", ""),
            title=payload.get("title", ""),
            pages=tuple(payload.get("pages", ())),
            text=payload.get("text", ""),
        )


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
    sections: tuple[Section, ...] = ()
    columns: dict = field(default_factory=dict)   # IMRaD 컬럼별 본문
    others: dict = field(default_factory=dict)    # 표준 구분에 없던 절

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
                by_path[path] = PaperRef(
                    id=paper_id, title=title, pdf_path=path, from_metadata=True
                )

        # 사용자가 직접 넣어둔 PDF도 잡는다. Agent 1 을 거치지 않고 파일만 떨어뜨리는
        # 경우가 있고, 그때 파일명은 보통 arXiv id 다.
        if self.pdf_dir.is_dir():
            for path in sorted(self.pdf_dir.glob("*.pdf")):
                by_path.setdefault(
                    path,
                    PaperRef(
                        id=path.stem, title=path.stem, pdf_path=path, from_metadata=False
                    ),
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
            numbered = self._refine(pages, with_tables)
            sections = self._build_sections(numbered)
            content = "\n\n".join(text for _, text in numbered)
        except PaperExtractionError:
            raise
        except Exception as exc:
            logger.log(LogCode.PAPER_EXTRACTION_FAILED, paper_id=paper_id, reason=str(exc))
            raise PaperExtractionError(f"'{paper_id}' 추출에 실패했습니다: {exc}") from exc

        # DB를 거치지 않은 PDF는 제목이 파일명이다. 그때만 PDF 안에 든 제목을 쓴다.
        # 제목과 파일명이 같다고 판단하면, 제목 그대로 저장된 PDF 에서 오판한다.
        title = ref.title
        if not ref.from_metadata and self._embedded_title:
            title = self._embedded_title

        extracted = ExtractionResult(
            id=ref.id,
            title=title,
            source_pdf=ref.pdf_path.name,
            content=content,
            n_pages=len(pages),
            n_vision_pages=n_vision,
            sections=sections,
        )

        # 마크다운을 먼저 쓴다. 뒤이은 분류는 모델을 부르므로 실패하거나 오래 걸릴 수
        # 있는데, 추출한 본문 자체는 그것과 무관하게 이미 완성돼 있다.
        self._write_markdown(extracted)

        # 절 이름은 논문마다 제각각이라, 표준 골격에 맞추는 일은 모델에게 맡긴다.
        mapping = self._classify_sections(title, sections)
        columns, others = self._to_imrad(sections, mapping)
        result = replace(extracted, columns=columns, others=others)

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

        # 동시요청 제한에 걸려 밀린 쪽은 병렬로 다시 보내도 같은 자리에서 또 밀린다.
        # 남은 쪽만 한 장씩 순서대로 보내면 제한에 걸리지 않아 대개 여기서 회복된다.
        # 폴백된 쪽은 2단 조판이 뒤섞여 대단원 표제가 본문 문장 사이로 끼어들므로,
        # 몇 초 더 쓰더라도 되살리는 편이 낫다. 전부 실패했다면 서비스가 내려간
        # 것이라 보고 다시 훑지 않는다.
        stalled = [index for index, (_, ok) in enumerate(outcomes) if not ok]
        if stalled and len(stalled) < len(images):
            for index in stalled:
                text, ok = read_one(index)
                if ok:
                    outcomes[index] = (text, True)

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

        # 비전 모델은 표제를 `**Abstract**` 처럼 굵은 글씨 한 줄로 내놓기도 한다.
        # 강조 기호를 벗겨야 아는 표제로 알아보고, 벗긴 뒤에도 아래 검사들을
        # 모두 통과해야 헤딩이 되므로 본문 강조가 표제로 둔갑하지는 않는다.
        stripped = stripped.strip("*_ ").strip()
        if not stripped:
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

    def _refine_page(
        self, page: str, running: set[str], with_tables: bool = True
    ) -> list[str]:
        """한 쪽을 읽을 만한 마크다운 블록 목록으로 정리한다.

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
                # 문단에 파묻힌 절 표제를 떼어낸다. 안 그러면 그 절이 앞 절에 흡수된다.
                refined.extend(self._split_inline_headings(paragraph))
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
        return refined

    def _refine(self, pages: list[str], with_tables: bool = True) -> list[tuple[int, str]]:
        """모든 쪽에 같은 규칙을 적용하고, 쪽번호를 붙인 블록 목록으로 돌려준다.

        여기서 쪽번호를 잃으면 심층 질의응답이 "3페이지의 수식" 같은 질문에 근거를
        짚을 수 없다. 하나의 마크다운으로 합치는 것은 호출한 쪽에서 한다.
        """
        running = self._running_lines(pages)
        numbered: list[tuple[int, str]] = []
        for page_number, page in enumerate(pages, start=1):
            for block in self._refine_page(page, running, with_tables):
                if block.strip():
                    numbered.append((page_number, block.strip()))
        return numbered

    # ── 대단원 스키마 ────────────────────────────────────────────────

    @staticmethod
    def _is_heading(text: str) -> bool:
        first = text.lstrip().splitlines()[0] if text.strip() else ""
        return first.startswith("#")

    @staticmethod
    def _split_section_label(heading_text: str) -> tuple[str, str]:
        """헤딩에서 (절 번호, 절 제목) 을 떼어낸다. 번호가 없으면 빈 문자열."""
        title = heading_text.lstrip("#").strip()
        matched = SECTION_LABEL_PATTERN.match(title)
        if not matched:
            return "", title
        return matched.group(1).rstrip(".").strip(), matched.group(2).strip()

    @classmethod
    def _split_inline_headings(cls, paragraph: str) -> list[str]:
        """문단에 파묻힌 절 표제를 떼어내 별도 헤딩 블록으로 만든다.

        표제가 늘 독립된 줄로 오지는 않는다. IEEE 양식은 초록 표제를 본문 첫 줄에
        붙여 쓰고, 폴백된 쪽에서는 절 표제가 문단 한가운데 섞인다. 그대로 두면
        그 절이 통째로 앞 절에 흡수되어 abstract·conclusion 컬럼이 빈다.
        """
        pieces: list[str] = []

        matched = INLINE_ABSTRACT_PATTERN.match(paragraph)
        if matched:
            pieces.append(f"## {matched.group(1).title()}")
            paragraph = paragraph[matched.end():].strip()
            if not paragraph:
                return pieces

        parts = INLINE_SECTION_PATTERN.split(paragraph)
        if len(parts) == 1:
            pieces.append(paragraph)
            return pieces

        # split 결과는 [앞글, 번호, 표제, 뒷글, 번호, 표제, ...] 로 이어진다.
        leading = parts[0].strip()
        if leading:
            pieces.append(leading)
        for index in range(1, len(parts), 3):
            number, heading, tail = parts[index], parts[index + 1], parts[index + 2]
            pieces.append(f"## {number} {heading.strip()}")
            if tail.strip():
                pieces.append(tail.strip())
        return pieces

    @staticmethod
    def _numbering_style(numbers: list[str]) -> str:
        """문서가 대단원에 어떤 번호 체계를 쓰는지 정한다. "arabic" | "roman" | "" 를 돌려준다.

        라벨 하나만 봐서는 못 가린다. ``C`` 는 로마 숫자 100 이면서 소단원 글자이기도
        해서, 실제로 ``C. Tasks`` 가 대단원으로 승격된 적이 있다. 문서 전체를 보면
        확정된다. ``II``, ``III`` 처럼 두 글자 이상인 로마 숫자가 있으면 그 문서는
        로마 숫자로 대단원을 매기는 것이고, 그때 한 글자짜리는 소단원 글자다.
        """
        if any(len(n) > 1 and ROMAN_SECTION_NUMBER.match(n) for n in numbers):
            return "roman"
        if any(ARABIC_SECTION_NUMBER.match(n) for n in numbers):
            return "arabic"
        return ""

    @staticmethod
    def _roman_value(number: str) -> int:
        """로마 숫자를 정수로. 로마 숫자가 아니면 0."""
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        if not number or any(ch not in values for ch in number):
            return 0
        total = 0
        for index, ch in enumerate(number):
            current = values[ch]
            following = values.get(number[index + 1], 0) if index + 1 < len(number) else 0
            total += -current if current < following else current
        return total

    @classmethod
    def _is_major_section(
        cls, number: str, title: str, style: str = "", last_major: int = 0
    ) -> bool:
        """대단원인지 판정한다.

        비전 모델이 매기는 헤딩 레벨은 들쭉날쭉해서(같은 논문에서 ``### 1 Introduction``
        과 ``## 3 Original Beam Search`` 가 섞인다) 레벨로는 가릴 수 없다. 문서의 번호
        체계와 절 이름으로 판단한다. 참고문헌은 대단원이어도 담지 않는다.

        로마 체계에서는 번호를 **순서**로 본다. ``C`` 는 로마 숫자 100 이라 라벨만으로는
        소단원 글자와 구별되지 않지만, 대단원은 I·II·III·IV·V·VI 로 이어지므로 앞
        대단원 다음 값인지 보면 갈린다. ``V. RESULTS`` 는 살리고 ``C. Tasks`` 는 거른다.
        """
        plain = title.strip().lower().rstrip(".")
        if plain in EXCLUDED_SECTION_TITLES:
            return False

        if number:
            if style == "arabic":
                return bool(ARABIC_SECTION_NUMBER.match(number))
            if style == "roman":
                value = cls._roman_value(number)
                return value > 0 and value == last_major + 1
            if ARABIC_SECTION_NUMBER.match(number) or ROMAN_SECTION_NUMBER.match(number):
                return True

        return plain in MAJOR_SECTION_TITLES

    def _build_sections(self, numbered: list[tuple[int, str]]) -> tuple[Section, ...]:
        """쪽번호가 붙은 블록을 논문의 대단원 단위로 묶는다.

        대단원 헤딩을 만나면 새 절을 열고, 그 뒤의 소단원과 본문은 열려 있는 절에
        이어 붙인다. 참고문헌 대단원부터는 담지 않는다. 첫 대단원 앞의 표지
        (제목·저자·소속)는 어느 절에도 넣지 않는다.
        """
        # 문서가 쓰는 번호 체계를 먼저 정한다. 라벨 하나만 봐서는 C 가 로마 숫자인지
        # 소단원 글자인지 못 가린다.
        style = self._numbering_style(
            [
                self._split_section_label(text)[0]
                for _, text in numbered
                if self._is_heading(text)
            ]
        )

        sections: list[Section] = []
        title = ""
        number = ""
        pages: list[int] = []
        parts: list[str] = []
        collecting = False
        stopped = False
        last_major = 0   # 로마 체계에서 대단원 번호가 이어지는지 보기 위해 든다

        def close() -> None:
            if collecting and parts:
                sections.append(
                    Section(
                        no=number,
                        title=title,
                        pages=tuple(sorted(set(pages))),
                        text="\n\n".join(parts).strip(),
                    )
                )

        for page, text in numbered:
            if self._is_heading(text):
                heading_no, heading_title = self._split_section_label(text)
                plain = heading_title.strip().lower().rstrip(".")

                if plain in EXCLUDED_SECTION_TITLES:
                    close()
                    collecting = False
                    stopped = True          # 참고문헌부터는 끝까지 담지 않는다
                    continue

                if not stopped and self._is_major_section(
                    heading_no, heading_title, style, last_major
                ):
                    close()
                    number, title = heading_no, heading_title
                    if style == "roman":
                        last_major = self._roman_value(heading_no) or last_major
                    pages, parts = [], []
                    collecting = True
                    continue

            if collecting:
                pages.append(page)
                parts.append(text)

        close()
        return tuple(sections)

    @staticmethod
    def _keyword_bucket(title: str) -> str:
        """절 이름의 낱말만 보고 구분을 고른다. 모델을 못 쓸 때의 대비책이다."""
        plain = title.strip().lower()
        for bucket, keywords in SECTION_KEYWORDS.items():
            if any(keyword in plain for keyword in keywords):
                return bucket
        return OTHERS_COLUMN

    def _classify_sections(self, title: str, sections: tuple[Section, ...]) -> dict[str, str]:
        """대단원을 IMRaD 구분으로 나눈다.

        절 이름은 논문마다 제각각이라('Original Beam Search', 'Direct Numerical
        Simulations') 규칙만으로는 못 가린다. 절 목록과 첫 문장만 모델에 넘겨
        한 번에 분류시킨다. 본문을 통째로 보내지 않으므로 호출 한 번이면 된다.

        모델을 못 부르거나 응답이 깨지면 낱말 규칙으로 되돌린다. 어느 경우에도
        분류에 실패한 절은 others 로 가므로 내용을 잃지 않는다.
        """
        fallback = {section.title: self._keyword_bucket(section.title) for section in sections}
        if not sections or not is_available():
            return fallback

        listing = "\n".join(
            f"{index}. {section.no + ' ' if section.no else ''}{section.title}\n"
            f"   opening: {' '.join(section.text.split())[:120]}"
            for index, section in enumerate(sections, start=1)
        )
        try:
            response = chat(
                [
                    {"role": "system", "content": SECTION_CLASSIFY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Paper title: {title}\n\n[Sections]\n{listing}",
                    },
                ],
                max_tokens=1024,
            )
        except NvidiaServiceError as exc:
            logger.log(LogCode.SECTION_CLASSIFY_FALLBACK, reason=str(exc)[:120])
            return fallback

        payload = self._parse_classification(response)
        if not payload:
            logger.log(LogCode.SECTION_CLASSIFY_FALLBACK, reason="응답을 해석하지 못함")
            return fallback

        # 모델은 목록에 보인 대로 번호를 붙여 답한다("3 Original Beam Search"). 제목만으로
        # 찾으면 못 맞춰 전부 대비책으로 떨어지므로, 번호를 떼고 맞춘다.
        answers = {
            self._normalize_label(key): str(value).strip().lower()
            for key, value in payload.items()
        }
        allowed = set(IMRAD_COLUMNS) | {OTHERS_COLUMN}

        mapping: dict[str, str] = {}
        for section in sections:
            bucket = answers.get(self._normalize_label(section.title), "")
            mapping[section.title] = bucket if bucket in allowed else fallback[section.title]
        return mapping

    @staticmethod
    def _normalize_label(label: str) -> str:
        """절 이름을 맞춰보기 좋게 다듬는다. 앞의 번호와 구두점을 떼고 소문자로."""
        stripped = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]{1,5}|[A-Z])[.)]?\s+", "", label.strip())
        return re.sub(r"[^a-z0-9 ]", "", stripped.lower()).strip()

    @staticmethod
    def _parse_classification(response: str) -> dict:
        """모델 응답에서 JSON 을 꺼낸다. 코드펜스나 군더더기가 붙어도 살려낸다."""
        text = response.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        if not text.startswith("{"):
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            text = brace.group(0) if brace else text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _to_imrad(
        self, sections: tuple[Section, ...], mapping: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """대단원을 IMRaD 컬럼별로 합친다. (컬럼값, others) 를 돌려준다.

        한 컬럼에 절이 여럿 들어갈 수 있으므로 절 제목을 헤딩으로 남긴다. 그래야
        합쳐진 뒤에도 논문이 원래 쓰던 절 이름을 알 수 있다.
        """
        buckets: dict[str, list[str]] = {name: [] for name in IMRAD_COLUMNS}
        others: dict[str, str] = {}

        for section in sections:
            label = f"{section.no} {section.title}".strip()
            body = f"## {label}\n\n{section.text}"
            target = mapping.get(section.title, OTHERS_COLUMN)
            if target in buckets:
                buckets[target].append(body)
            else:
                others[label] = section.text

        return {name: "\n\n".join(parts) for name, parts in buckets.items()}, others

    def get_part(self, paper_id: str, name: str) -> str:
        """심층 질의응답이 쓰는 진입점: 표준 구분 하나의 본문을 돌려준다.

        ``get_part(pid, "method")`` 처럼 부른다. 논문이 그 구분에 해당하는 절을
        갖고 있지 않으면 빈 문자열이다.
        """
        record = self.get(paper_id)
        if not record or name not in IMRAD_COLUMNS:
            return ""
        return record.get(name) or ""

    def get_parts(self, paper_id: str) -> dict[str, str]:
        """표준 구분 전체를 돌려준다. 내용이 있는 것만 담는다."""
        record = self.get(paper_id)
        if not record:
            return {}
        return {
            name: record[name]
            for name in IMRAD_COLUMNS
            if record.get(name)
        }

    def get_others(self, paper_id: str) -> dict[str, str]:
        """표준 구분에 넣지 못한 절. {절 이름: 내용} 형태다."""
        record = self.get(paper_id)
        if not record:
            return {}
        try:
            payload = json.loads(record.get("others") or "{}")
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    # ── 저장 ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extracted (
                    id           TEXT PRIMARY KEY,
                    title        TEXT,
                    source_pdf   TEXT,
                    abstract     TEXT,
                    introduction TEXT,
                    related_work TEXT,
                    method       TEXT,
                    experiment   TEXT,
                    result       TEXT,
                    conclusion   TEXT,
                    others       TEXT,
                    content      TEXT,
                    n_pages      INTEGER,
                    n_chars      INTEGER,
                    extractor    TEXT,
                    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def markdown_path(self, paper_id: str) -> Path:
        """사람이 바로 열어볼 수 있는 마크다운 파일 경로."""
        return self.output_dir / f"{paper_id}.md"

    def _write_markdown(self, result: ExtractionResult) -> Path:
        """가공본을 마크다운 파일로도 떨어뜨린다. 앞머리에 출처를 적어 둔다.

        DB보다 먼저 불릴 수 있으므로 폴더는 여기서도 만든다.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        """가공본을 DB에 넣고 제목 목록 JSON 을 갱신한다.

        마크다운은 추출 직후에 이미 써 두었으므로 여기서 다시 쓰지 않는다.
        """
        self._init_db()
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO extracted "
                "(id, title, source_pdf, abstract, introduction, related_work, method, "
                "experiment, result, conclusion, others, content, n_pages, n_chars, "
                "extractor, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (
                    result.id,
                    result.title,
                    result.source_pdf,
                    *[result.columns.get(name, "") for name in IMRAD_COLUMNS],
                    json.dumps(result.others, ensure_ascii=False),
                    result.content,
                    result.n_pages,
                    result.n_chars,
                    result.extractor,
                ),
            )
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

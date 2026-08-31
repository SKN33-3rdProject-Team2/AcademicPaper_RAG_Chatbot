"""번역·요약에 사용하는 학술 Markdown 분리와 청킹을 담당한다."""

from __future__ import annotations

import re
from dataclasses import dataclass


REFERENCE_HEADING_PATTERN = re.compile(
    r"^#{1,6}\s*(references|bibliography)\s*$", re.IGNORECASE
)
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
LATEX_ENV_PATTERN = re.compile(r"\\(begin|end)\{([^{}]+)\}")
PROTECTED_LATEX_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "aligned",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "cases",
    "split",
    "array",
    "matrix",
    "pmatrix",
    "bmatrix",
    "Bmatrix",
    "vmatrix",
    "Vmatrix",
    "tabular",
    "tabular*",
}
_LATEX_ENVIRONMENTS_PATTERN = "|".join(
    sorted((re.escape(name) for name in PROTECTED_LATEX_ENVIRONMENTS), key=len, reverse=True)
)
LATEX_ENV_BLOCK_PATTERN = re.compile(
    rf"\\begin\{{({_LATEX_ENVIRONMENTS_PATTERN})\}}.*?\\end\{{\1\}}",
    re.DOTALL,
)
DISPLAY_MATH_PATTERN = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
INLINE_MATH_PATTERN = re.compile(
    r"(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$|\\\(.*?\\\)",
    re.DOTALL,
)
HTML_TABLE_TAG_PATTERN = re.compile(
    r"</?(?:table|thead|tbody|tfoot|tr|th|td)(?:\s[^>]*)?>",
    re.IGNORECASE,
)
HTML_TABLE_BLOCK_PATTERN = re.compile(
    r"<table(?:\s[^>]*)?>.*?</table>",
    re.IGNORECASE | re.DOTALL,
)
PROTECTED_TOKEN_PATTERN = re.compile(r"__APRAG_PROTECTED_[0-9]{6}__")


class TranslationMarkupError(RuntimeError):
    """번역 모델이 보호된 수식 또는 표 구조를 훼손했을 때 발생한다."""

    reason = "protected_markup_changed"
    retryable = True
    status_code = None


@dataclass(frozen=True)
class ProtectedTranslationMarkup:
    """모델 입력용 본문과 원문 복원에 필요한 보호 정보."""

    text: str
    replacements: dict[str, str]
    token_order: tuple[str, ...]


def strip_metadata_header(markdown: str) -> str:
    """도구가 붙인 문서 메타데이터 헤더만 제거한다."""
    lines = markdown.splitlines()
    for index, line in enumerate(lines[:12]):
        if line.strip() != "---":
            continue
        header = "\n".join(lines[:index])
        if "**arXiv ID**" in header or "**번역 모델**" in header:
            return "\n".join(lines[index + 1 :]).lstrip("\n")
        break
    return markdown


def split_reference_section(markdown: str) -> tuple[str, str]:
    """본문과 참고문헌을 분리한다."""
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if REFERENCE_HEADING_PATTERN.match(line.strip()):
            return "\n".join(lines[:index]), "\n".join(lines[index:])
    return markdown, ""


def _paragraph_blocks(markdown: str) -> list[str]:
    """빈 줄이 포함된 수식·HTML 표를 하나의 문단 블록으로 묶는다."""
    blocks: list[str] = []
    buffer: list[str] = []
    in_display_math = False
    in_bracket_math = False
    in_html_table = False
    environment_depth: dict[str, int] = {}
    for paragraph in markdown.split("\n\n"):
        buffer.append(paragraph)
        if paragraph.count("$$") % 2 == 1:
            in_display_math = not in_display_math
        if paragraph.count(r"\[") > paragraph.count(r"\]"):
            in_bracket_math = True
        elif paragraph.count(r"\]") > paragraph.count(r"\["):
            in_bracket_math = False

        normalized = paragraph.casefold()
        if normalized.count("<table") > normalized.count("</table>"):
            in_html_table = True
        elif normalized.count("</table>") > normalized.count("<table"):
            in_html_table = False

        for match in LATEX_ENV_PATTERN.finditer(paragraph):
            action, environment = match.groups()
            if environment not in PROTECTED_LATEX_ENVIRONMENTS:
                continue
            delta = 1 if action == "begin" else -1
            environment_depth[environment] = max(
                0, environment_depth.get(environment, 0) + delta
            )

        if not (
            in_display_math
            or in_bracket_math
            or in_html_table
            or any(environment_depth.values())
        ):
            blocks.append("\n\n".join(buffer))
            buffer.clear()
    if buffer:
        blocks.append("\n\n".join(buffer))
    return blocks


def _is_markdown_table(block: str) -> bool:
    """마크다운 표의 헤더 구분 행이 있는지 확인한다."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2 and all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in cells
        ):
            return any("|" in candidate for candidate in lines if candidate != line)
    return False


def extract_markdown_tables(markdown: str) -> list[str]:
    """본문에서 완전한 Markdown 표 블록을 원문 순서대로 반환한다."""
    return [block for block in _paragraph_blocks(markdown) if _is_markdown_table(block)]


def _markdown_row_cells(line: str) -> tuple[list[str], bool, bool, str]:
    """Markdown 표 행의 셀과 바깥 파이프·들여쓰기를 분리한다."""
    indent = line[: len(line) - len(line.lstrip())]
    content = line.strip()
    leading_pipe = content.startswith("|")
    trailing_pipe = content.endswith("|") and not content.endswith(r"\|")
    if leading_pipe:
        content = content[1:]
    if trailing_pipe:
        content = content[:-1]
    cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", content)]
    return cells, leading_pipe, trailing_pipe, indent


def _is_markdown_separator_row(line: str) -> bool:
    cells, _leading, _trailing, _indent = _markdown_row_cells(line)
    return len(cells) >= 2 and all(
        re.fullmatch(r":?-{3,}:?", cell) for cell in cells
    )


def rebuild_markdown_table(original: str, translated: str) -> str:
    """번역된 셀 텍스트를 원래 Markdown 표의 행·열 구조에 맞춰 재조립한다."""
    original_lines = [line for line in original.splitlines() if line.strip()]
    translated_lines = [
        line
        for line in translated.strip().splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    if len(original_lines) != len(translated_lines):
        raise TranslationMarkupError("번역된 표의 행 개수가 원문과 다릅니다.")

    rebuilt: list[str] = []
    for original_line, translated_line in zip(original_lines, translated_lines):
        if _is_markdown_separator_row(original_line):
            rebuilt.append(original_line)
            continue

        original_cells, leading, trailing, indent = _markdown_row_cells(original_line)
        translated_cells, _translated_leading, _translated_trailing, _ = (
            _markdown_row_cells(translated_line)
        )
        if len(original_cells) != len(translated_cells):
            raise TranslationMarkupError("번역된 표의 열 개수가 원문과 다릅니다.")

        row = " | ".join(translated_cells)
        if leading:
            row = f"| {row}"
        if trailing:
            row = f"{row} |"
        rebuilt.append(f"{indent}{row}")

    result = "\n".join(rebuilt)
    if not _is_markdown_table(result):
        raise TranslationMarkupError("번역된 표를 Markdown 표로 복원하지 못했습니다.")
    return result


def _is_protected_block(block: str) -> bool:
    """청크 경계에서 분리하면 안 되는 수식·표 블록인지 확인한다."""
    normalized = block.casefold()
    has_protected_environment = any(
        environment in PROTECTED_LATEX_ENVIRONMENTS
        for _action, environment in LATEX_ENV_PATTERN.findall(block)
    )
    return (
        "$$" in block
        or r"\[" in block
        or r"\]" in block
        or has_protected_environment
        or "<table" in normalized
        or "</table>" in normalized
        or _is_markdown_table(block)
    )


def _split_oversized_text(block: str, max_chars: int) -> list[str]:
    """보호 블록을 제외한 긴 문단을 줄·문자 경계에서 나눈다."""
    if len(block) <= max_chars or _is_protected_block(block):
        return [block]

    pieces: list[str] = []
    current = ""
    for line in block.splitlines(keepends=True):
        if current and len(current) + len(line) > max_chars:
            pieces.append(current.rstrip("\n"))
            current = ""
        while len(line) > max_chars:
            room = max_chars - len(current)
            current += line[:room]
            pieces.append(current.rstrip("\n"))
            current = ""
            line = line[room:]
        current += line
    if current:
        pieces.append(current.rstrip("\n"))
    return [piece for piece in pieces if piece]


def split_markdown(markdown: str, *, max_chars: int) -> list[str]:
    """문단 경계와 수식·표 블록을 보존하며 마크다운을 청크로 나눈다."""
    if max_chars < 1:
        raise ValueError("max_chars는 1 이상이어야 합니다.")

    blocks = [
        piece
        for block in _paragraph_blocks(markdown)
        for piece in _split_oversized_text(block, max_chars)
        if piece.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        separator_size = 2 if current else 0
        if current and size + separator_size + len(block) > max_chars:
            chunks.append("\n\n".join(current))
            current, size = [], 0
            separator_size = 0
        current.append(block)
        size += separator_size + len(block)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _markdown_sections(markdown: str) -> list[tuple[str, list[str]]]:
    """Markdown 헤딩 계층과 그 아래 문단들을 논리 섹션으로 묶는다."""
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[str] = []
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            sections.append(("\n".join(heading_stack), paragraphs.copy()))
            paragraphs.clear()

    for block in _paragraph_blocks(markdown):
        stripped = block.strip()
        heading = MARKDOWN_HEADING_PATTERN.fullmatch(stripped)
        if heading and "\n" not in stripped:
            flush()
            level = len(heading.group(1))
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(stripped)
            continue
        if stripped:
            paragraphs.append(block)
    flush()
    return sections


def split_markdown_by_section(markdown: str, *, max_chars: int) -> list[str]:
    """헤딩 계층·문단·수식·표를 보존하여 요약용 청크를 만든다.

    한 섹션이 여러 청크로 나뉘면 현재 헤딩 경로를 각 청크 앞에 반복한다.
    따라서 모델은 청크만 받아도 해당 문단의 논문 내 문맥을 알 수 있다.
    """
    if max_chars < 1:
        raise ValueError("max_chars는 1 이상이어야 합니다.")

    section_chunks: list[str] = []
    for heading_context, paragraphs in _markdown_sections(markdown):
        prefix = heading_context.strip()
        available_chars = max(1, max_chars - len(prefix) - (2 if prefix else 0))
        pieces = [
            piece
            for paragraph in paragraphs
            for piece in _split_oversized_text(paragraph, available_chars)
            if piece.strip()
        ]
        current: list[str] = [prefix] if prefix else []
        current_size = len(prefix)
        content_count = 0
        for piece in pieces:
            separator_size = 2 if current else 0
            if content_count and current_size + separator_size + len(piece) > max_chars:
                section_chunks.append("\n\n".join(current))
                current = [prefix] if prefix else []
                current_size = len(prefix)
                content_count = 0
                separator_size = 2 if current else 0
            current.append(piece)
            current_size += separator_size + len(piece)
            content_count += 1
        if content_count:
            section_chunks.append("\n\n".join(current))

    # 서로 인접한 짧은 섹션은 제한 안에서 합쳐 호출 횟수를 줄인다.
    chunks: list[str] = []
    current_sections: list[str] = []
    current_size = 0
    for section in section_chunks:
        separator_size = 2 if current_sections else 0
        if current_sections and current_size + separator_size + len(section) > max_chars:
            chunks.append("\n\n".join(current_sections))
            current_sections = []
            current_size = 0
            separator_size = 0
        current_sections.append(section)
        current_size += separator_size + len(section)
    if current_sections:
        chunks.append("\n\n".join(current_sections))
    return chunks


def protect_translation_markup(
    markdown: str, *, protect_tables: bool = True
) -> ProtectedTranslationMarkup:
    """LaTeX와 표 블록을 고유 토큰으로 바꿔 모델의 변경을 막는다.

    일반 본문 번역에서는 표 전체를 보호한다. 표 전용 번역에서는
    ``protect_tables=False``로 두고 LaTeX와 HTML 태그만 보호한다.
    """
    if PROTECTED_TOKEN_PATTERN.search(markdown):
        raise TranslationMarkupError(
            "원문에 번역 보호 토큰과 충돌하는 문자열이 포함되어 있습니다."
        )

    replacements: dict[str, str] = {}

    def token_for(original: str) -> str:
        token = f"__APRAG_PROTECTED_{len(replacements) + 1:06d}__"
        replacements[token] = original
        return token

    protected = markdown
    if protect_tables:
        protected = HTML_TABLE_BLOCK_PATTERN.sub(
            lambda match: token_for(match.group(0)), protected
        )
        for table in extract_markdown_tables(protected):
            protected = protected.replace(table, token_for(table), 1)

    protected = LATEX_ENV_BLOCK_PATTERN.sub(
        lambda match: token_for(match.group(0)), protected
    )
    protected = DISPLAY_MATH_PATTERN.sub(
        lambda match: token_for(match.group(0)), protected
    )
    protected = INLINE_MATH_PATTERN.sub(
        lambda match: token_for(match.group(0)), protected
    )

    if not protect_tables:
        protected = HTML_TABLE_TAG_PATTERN.sub(
            lambda match: token_for(match.group(0)), protected
        )
    return ProtectedTranslationMarkup(
        text=protected,
        replacements=replacements,
        token_order=tuple(PROTECTED_TOKEN_PATTERN.findall(protected)),
    )


def restore_translation_markup(
    translated: str, protection: ProtectedTranslationMarkup
) -> str:
    """보호 토큰의 누락·중복·순서 변경을 검사하고 원문을 복원한다."""
    actual_order = tuple(PROTECTED_TOKEN_PATTERN.findall(translated))
    if actual_order != protection.token_order:
        raise TranslationMarkupError(
            "번역 과정에서 LaTeX 또는 표 구조 보호 토큰이 변경되었습니다."
        )
    restored = translated
    for token, original in protection.replacements.items():
        restored = restored.replace(token, original)
    if PROTECTED_TOKEN_PATTERN.search(restored):
        raise TranslationMarkupError("번역 보호 토큰을 완전히 복원하지 못했습니다.")
    return restored

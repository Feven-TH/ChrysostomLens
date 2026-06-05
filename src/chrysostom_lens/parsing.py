"""Step 1 and Step 2: PDF structural parsing and homily-aware batching."""

from __future__ import annotations

import json
import re
from statistics import median
from pathlib import Path
from typing import Iterable, TypedDict

import pdfplumber

from chrysostom_lens.models import ParagraphBatch, ParsedParagraph

HOMILY_RE = re.compile(
    r"^\s*(?:homily|homil\.)\s+([ivxlcdm]+|\d+)\b\.?",
    flags=re.IGNORECASE,
)
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
LEADING_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}\s+(?=[A-Za-z“\"'])")
TERMINAL_RE = re.compile(r"[.!?][”’\"')\]]?$")
LOWERCASE_CONTINUATION_RE = re.compile(r"^[a-z]")
SCRIPTURE_REFERENCE_RE = re.compile(r"^MATTHEW\s+\d+:\d+$", flags=re.IGNORECASE)
VERSE_ID_RE = re.compile(r"^<(\d+)>$")


class ExtractedLine(TypedDict):
    text: str
    top: float | None


def _canonical_homily(line: str) -> str | None:
    match = HOMILY_RE.match(line)
    if not match:
        return None
    return f"HOMILY {match.group(1).upper()}"


def _clean_paragraph(lines: Iterable[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return re.sub(r"\s+", " ", text).strip()


def _strip_page_number(text: str) -> str:
    text = text.strip()
    if PAGE_NUMBER_RE.fullmatch(text):
        return ""
    return LEADING_PAGE_NUMBER_RE.sub("", text).strip()


def _should_merge_with_previous(previous: str, current: str) -> bool:
    """Return True when a PDF block is almost certainly a paragraph continuation."""

    previous = previous.rstrip()
    current = current.lstrip()
    if not previous or not current:
        return False

    if LOWERCASE_CONTINUATION_RE.match(current):
        return True
    if previous.endswith((",", ";", ":", "—", "-", "–")):
        return True
    if not TERMINAL_RE.search(previous):
        return True
    return False


def _has_same_context(left: ParsedParagraph, right: ParsedParagraph) -> bool:
    return (
        left.homily == right.homily
        and left.scripture_reference == right.scripture_reference
        and left.verse_id == right.verse_id
    )


def _extract_page_lines(page) -> list[ExtractedLine]:
    """Return layout-preserving page lines with approximate vertical positions."""

    if hasattr(page, "extract_text_lines"):
        try:
            extracted = page.extract_text_lines(layout=True, strip=False, return_chars=False) or []
            return [
                {"text": str(line.get("text", "")), "top": float(line["top"]) if line.get("top") is not None else None}
                for line in extracted
            ]
        except TypeError:
            pass

    text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or ""
    return [{"text": raw_line, "top": None} for raw_line in text.splitlines()]


def _page_median_gap(lines: list[ExtractedLine]) -> float | None:
    tops = [line["top"] for line in lines if line["top"] is not None and line["text"].strip()]
    gaps = [
        current - previous
        for previous, current in zip(tops, tops[1:])
        if current > previous
    ]
    return median(gaps) if gaps else None


def _reconstruct_paragraphs(blocks: list[ParsedParagraph]) -> list[ParsedParagraph]:
    reconstructed: list[ParsedParagraph] = []

    for block in blocks:
        text = _strip_page_number(block.paragraph_text)
        if not text:
            continue

        if (
            reconstructed
            and _has_same_context(reconstructed[-1], block)
            and _should_merge_with_previous(reconstructed[-1].paragraph_text, text)
        ):
            reconstructed[-1] = ParsedParagraph(
                homily=reconstructed[-1].homily,
                paragraph_text=_clean_paragraph([reconstructed[-1].paragraph_text, text]),
                scripture_reference=reconstructed[-1].scripture_reference,
                verse_id=reconstructed[-1].verse_id,
            )
            continue

        reconstructed.append(
            ParsedParagraph(
                homily=block.homily,
                paragraph_text=text,
                scripture_reference=block.scripture_reference,
                verse_id=block.verse_id,
            )
        )

    return reconstructed


def parse_pdf_to_paragraphs(pdf_path: str) -> list[dict]:
    """Extract natural paragraphs from a PDF while continuously tracking homilies.

    Paragraph breaks are inferred from explicit blank extracted-text lines. Homily
    headings such as ``HOMILY I``, ``Homily 2``, and ``HOMILY XX`` update the
    active homily and are not included in paragraph text.
    """

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")

    blocks: list[ParsedParagraph] = []
    current_homily: str | None = None
    current_scripture_reference: str | None = None
    current_verse_id: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        paragraph = _clean_paragraph(current_lines)
        if paragraph and current_homily:
            blocks.append(
                ParsedParagraph(
                    homily=current_homily,
                    paragraph_text=paragraph,
                    scripture_reference=current_scripture_reference,
                    verse_id=current_verse_id,
                )
            )
        current_lines = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted_lines = _extract_page_lines(page)
            if not any(line["text"].strip() for line in extracted_lines):
                continue

            median_gap = _page_median_gap(extracted_lines)
            previous_top: float | None = None

            for extracted_line in extracted_lines:
                raw_line = extracted_line["text"]
                line = raw_line.strip()
                top = extracted_line["top"]
                large_gap = (
                    bool(line)
                    and previous_top is not None
                    and top is not None
                    and median_gap is not None
                    and top - previous_top > median_gap * 1.5
                )

                if large_gap and len(current_lines) >= 3:
                    flush()

                if line:
                    previous_top = top

                if PAGE_NUMBER_RE.fullmatch(line):
                    continue

                homily = _canonical_homily(line)
                if homily:
                    flush()
                    current_homily = homily
                    remainder = line[HOMILY_RE.match(line).end() :].strip(" .:-")
                    if remainder:
                        current_lines.append(remainder)
                    continue

                if SCRIPTURE_REFERENCE_RE.fullmatch(line):
                    flush()
                    current_scripture_reference = line
                    current_verse_id = None
                    continue

                verse_match = VERSE_ID_RE.fullmatch(line)
                if verse_match:
                    flush()
                    current_verse_id = verse_match.group(1)
                    continue

                if line == "":
                    flush()
                    continue

                current_lines.append(line)

    flush()

    paragraphs = _reconstruct_paragraphs(blocks)
    return [paragraph.model_dump() for paragraph in paragraphs]


def batch_paragraphs(
    paragraphs: list[dict] | list[ParsedParagraph],
    batch_size: int = 8,
) -> list[ParagraphBatch]:
    """Group consecutive paragraphs into windows of up to 8 without crossing homilies."""

    if batch_size != 8:
        raise ValueError("This pipeline requires operational windows of exactly 8 paragraphs.")

    parsed = [p if isinstance(p, ParsedParagraph) else ParsedParagraph.model_validate(p) for p in paragraphs]
    batches: list[ParagraphBatch] = []
    window: list[ParsedParagraph] = []
    window_start = 0
    active_homily: str | None = None

    for absolute_index, paragraph in enumerate(parsed):
        homily_changed = active_homily is not None and paragraph.homily != active_homily
        if window and (homily_changed or len(window) == batch_size):
            batches.append(
                ParagraphBatch(
                    homily=active_homily or window[0].homily,
                    start_paragraph=window_start,
                    paragraphs=window,
                )
            )
            window = []
            window_start = absolute_index

        if not window:
            active_homily = paragraph.homily
            window_start = absolute_index
        window.append(paragraph)

    if window:
        batches.append(
            ParagraphBatch(
                homily=active_homily or window[0].homily,
                start_paragraph=window_start,
                paragraphs=window,
            )
        )

    return batches


def save_parsed_cache(paragraphs: list[dict], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding="utf-8")


def load_parsed_cache(cache_path: Path) -> list[dict]:
    return json.loads(cache_path.read_text(encoding="utf-8"))

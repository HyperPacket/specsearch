import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from specsearch import config

# CSI section numbers like "03 30 00" — primary section breaks.
_CSI_RE = re.compile(r"^\d{2}\s+\d{2}\s+\d{2}")

# PART markers, numbered subsections, all-caps titles — minor breaks within a CSI section.
_MINOR_HEADING_RE = re.compile(
    r"^(?:"
    r"PART\s+\d+"                # PART 1, PART 2
    r"|\d+\.\d[\d.]*\s+\S"      # 1.1 SUMMARY, 2.3.1 FOO
    r"|[A-Z][A-Z\s\-/,]{3,60}$" # ALL-CAPS TITLE LINE
    r")"
)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page: int
    section: str


def _chunk_id(source_file: str, page: int, section: str, idx: int, text: str = "") -> str:
    key = f"{source_file}|{page}|{section}|{idx}|{text}"
    return hashlib.md5(key.encode()).hexdigest()  # md5 fine for non-crypto ID


def _sliding_window(words: list[str], size: int, overlap: int) -> list[str]:
    """Return overlapping text windows from a word list."""
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    result: list[str] = []
    for start in range(0, len(words), step):
        result.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return result


def _extract_sections(doc: fitz.Document, source_file: str) -> list[Chunk]:
    """Walk pages, detect headings, flush section word buffers into chunks.

    CSI section numbers (03 30 00) set the major_section context.
    PART markers and numbered subsections create minor breaks within that context,
    producing section labels like "03 30 00 CAST-IN-PLACE CONCRETE / 1.2 SUBMITTALS".
    Every chunk's section therefore traces back to its CSI division.
    """
    chunks: list[Chunk] = []
    major_section = "preamble"
    current_section = "preamble"
    current_page = 1
    section_words: list[str] = []

    def flush() -> None:
        if not section_words:
            return
        for idx, text in enumerate(
            _sliding_window(section_words, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_file, current_page, current_section, idx, text),
                    text=text,
                    source_file=source_file,
                    page=current_page,
                    section=current_section,
                )
            )

    for page_num, page in enumerate(doc, start=1):
        for line in page.get_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _CSI_RE.match(stripped):
                flush()
                section_words.clear()
                major_section = stripped[:80]
                current_section = major_section
                current_page = page_num
            elif _MINOR_HEADING_RE.match(stripped) and len(stripped) > 4:
                flush()
                section_words.clear()
                current_section = f"{major_section} / {stripped[:60]}"
                current_page = page_num
            else:
                section_words.extend(stripped.split())

    flush()
    return chunks


def ingest_pdf(pdf_path: Path) -> list[Chunk]:
    """Parse a single PDF and return its chunks."""
    doc = fitz.open(str(pdf_path))
    return _extract_sections(doc, pdf_path.name)


def ingest_directory(directory: Path) -> list[Chunk]:
    """Ingest all PDFs in a directory, sorted by name."""
    chunks: list[Chunk] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        chunks.extend(ingest_pdf(pdf_path))
    return chunks

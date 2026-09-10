"""
Stage 2 of the pipeline: PDF bytes -> text, one page at a time.

The one rule of this module: text is extracted PER PAGE and never concatenated
into a single blob. Page numbers are the backbone of citations - "every answer
cites the exact document and page" - and once pages are merged, the page a
sentence came from cannot be recovered.

Text extraction uses PyMuPDF. Table extraction (pdfplumber) is Phase 5; see the
stub at the bottom for why it is not simply "more parsing".
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf  # the modern import name; older tutorials use `import fitz`

# Pages with almost no text are skipped. Filings are full of cover pages, blank
# separators and full-page images; indexing those adds noise to retrieval and
# gives the model nothing to cite. 50 characters is a deliberately low bar - it
# drops the empties without discarding short but real pages.
MIN_PAGE_CHARS = 50


@dataclass(frozen=True)
class ParsedPage:
    """The text of a single PDF page.

    `page_number` is 1-based to match what a PDF reader shows the user, even
    though PyMuPDF indexes from 0. Converting once, here, avoids off-by-one
    citations pointing a reader at the wrong page.
    """

    page_number: int
    text: str


def parse_pdf(pdf_path: Path) -> list[ParsedPage]:
    """Extract text from a PDF, one entry per page with usable content."""
    pages: list[ParsedPage] = []

    # `with` closes the file handle even if extraction raises - these are large
    # files and ingestion opens many of them in one run.
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text")

            # Normalise whitespace. PDF text extraction routinely emits runs of
            # spaces and newlines from the original layout; left alone these
            # waste chunk budget and add nothing to meaning.
            text = "\n".join(
                line.strip() for line in text.splitlines() if line.strip()
            )

            if len(text) < MIN_PAGE_CHARS:
                continue

            pages.append(ParsedPage(page_number=page_index + 1, text=text))

    return pages


def page_count(pdf_path: Path) -> int:
    """Total pages in a PDF, including ones parse_pdf() skips.

    Useful for reporting coverage during ingestion: "480 of 512 pages had text"
    is a cheap, honest signal that extraction worked.
    """
    with pymupdf.open(pdf_path) as document:
        return document.page_count


# --- Phase 5: tables ---------------------------------------------------------
#
# Not implemented yet, and deliberately not a quick win.
#
# Financial tables are the numeric backbone of this corpus, but feeding a
# flattened table into a text chunk is worse than useless - the model reads
# stray numbers with no reliable row/column labels and confidently misreports
# them. project.md commits to the alternative: extract tables into Postgres and
# answer numeric questions with SQL, never by having the LLM do arithmetic over
# retrieved text.
#
# So this stays a stub until Phase 5 can do it properly (pdfplumber's
# `page.extract_tables()`, then normalising metric names across companies).
# Phase 1 answers narrative questions and is honest about the rest.

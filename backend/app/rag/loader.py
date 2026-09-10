"""
Stage 1 of the pipeline: find PDFs on disk and work out what each one IS.

The corpus is organised so that a file's path carries its metadata:

    data/raw/{company}/{fiscal_year}/{doc_type}.pdf
    data/raw/tcs/FY24/annual_report.pdf

Deriving metadata from the path keeps ingestion dependency-free and makes the
corpus self-describing. Richer fields that a path cannot express - full company
name, sector, the URL the PDF came from - live in data/manifest.csv and are
merged in here.

Design choice: this module FAILS LOUDLY on a path it cannot parse, rather than
skipping the file. A silently skipped filing looks identical to a filing that
contains no answer, which would be a genuinely nasty bug to track down later.
"""

import csv
from pathlib import Path

from app.config import MANIFEST_PATH, RAW_DIR
from app.schemas import DocumentMeta

# The document types the project covers. Kept as a closed set so a typo in a
# filename ("annual_reprot.pdf") is caught at ingestion instead of producing a
# phantom document type that no metadata filter will ever match.
DOC_TYPES = {"annual_report", "quarterly_results", "earnings_call"}


class CorpusLayoutError(Exception):
    """Raised when a file under data/raw/ does not match the expected layout."""


def parse_pdf_path(pdf_path: Path) -> DocumentMeta:
    """Turn a PDF's path into a DocumentMeta.

    Expects <RAW_DIR>/{company}/{fiscal_year}/{doc_type}.pdf.
    Raises CorpusLayoutError with an actionable message if it does not match.
    """
    try:
        relative = pdf_path.relative_to(RAW_DIR)
    except ValueError:
        raise CorpusLayoutError(
            f"{pdf_path} is not inside the corpus directory {RAW_DIR}"
        ) from None

    # Expect exactly three components: company / fiscal_year / filename.pdf
    if len(relative.parts) != 3:
        raise CorpusLayoutError(
            f"Expected {{company}}/{{fiscal_year}}/{{doc_type}}.pdf but got "
            f"{relative.as_posix()!r} ({len(relative.parts)} path components, "
            "expected 3). Example: tcs/FY24/annual_report.pdf"
        )

    company, fiscal_year, filename = relative.parts
    doc_type = Path(filename).stem

    if doc_type not in DOC_TYPES:
        raise CorpusLayoutError(
            f"{relative.as_posix()!r} has document type {doc_type!r}, which is not "
            f"one of {sorted(DOC_TYPES)}. Rename the file, or add the new type to "
            "DOC_TYPES in backend/app/rag/loader.py."
        )

    return DocumentMeta(
        company=company.lower(),
        fiscal_year=fiscal_year.upper(),
        doc_type=doc_type,
        # Stored with forward slashes so the value is identical on Windows and
        # Linux. It ends up in Qdrant payloads, so it must not vary by machine.
        source_path=relative.as_posix(),
    )


def load_manifest() -> dict[str, dict[str, str]]:
    """Read data/manifest.csv, keyed by source_path.

    The manifest is optional: without it the pipeline still works, and documents
    simply carry no company_name / sector / source_url. That keeps the barrier to
    dropping in a new PDF low.
    """
    if not MANIFEST_PATH.exists():
        return {}

    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        # Skip blank rows, which spreadsheet editors love to leave at the end.
        return {
            row["source_path"]: row
            for row in csv.DictReader(handle)
            if row.get("source_path")
        }


def discover_documents() -> list[DocumentMeta]:
    """Find every PDF in the corpus and return its metadata.

    Sorted so that ingestion order - and therefore the log output you read while
    debugging - is stable between runs.
    """
    if not RAW_DIR.exists():
        raise CorpusLayoutError(
            f"Corpus directory {RAW_DIR} does not exist. Create it and add PDFs as "
            "data/raw/{company}/{fiscal_year}/{doc_type}.pdf"
        )

    manifest = load_manifest()
    documents: list[DocumentMeta] = []

    for pdf_path in sorted(RAW_DIR.rglob("*.pdf")):
        meta = parse_pdf_path(pdf_path)

        # Merge in the manifest row, if this document has one.
        row = manifest.get(meta.source_path)
        if row:
            meta = meta.model_copy(
                update={
                    "company_name": row.get("company_name") or None,
                    "sector": row.get("sector") or None,
                    "source_url": row.get("source_url") or None,
                }
            )
        documents.append(meta)

    return documents


def absolute_path(meta: DocumentMeta) -> Path:
    """Resolve a DocumentMeta back to a readable path on disk."""
    return RAW_DIR / meta.source_path

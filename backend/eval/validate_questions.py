"""
Check that the eval question set is still valid against the corpus on disk.

This runs no model and costs no tokens. Its job is to catch label rot - the
failure where the question set silently stops describing the corpus it is
meant to measure:

  * a gold page that `parse.py` DISCARDS (under MIN_PAGE_CHARS) can never be
    retrieved, so a question pointing at it scores 0 forever and looks like a
    retrieval failure rather than a labelling mistake. This is the check that
    matters most, and it is invisible without parsing the PDFs.
  * a gold page past the end of its document, or a document that has been
    removed from data/raw/, means the label was written against a different
    copy of the filing.
  * a refusal question that carries gold pages is self-contradictory: the whole
    point is that the answer is NOT in the corpus.

Run it before every eval run, and after any change to chunking or parsing:

    python backend/eval/validate_questions.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import RAW_DIR  # noqa: E402
from app.rag.parse import page_count, parse_pdf  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"

CATEGORIES = {
    "lookup",
    "narrative",
    "multi_year",
    "cross_document",
    "arithmetic_boundary",
    "refusal",
}
DIFFICULTIES = {"easy", "medium", "hard"}
REFUSAL_KINDS = {"insufficient_context", "out_of_scope"}


def load_questions() -> list[dict]:
    """Read the JSONL question set, failing loudly on a malformed line."""
    questions = []
    with QUESTIONS_PATH.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{QUESTIONS_PATH.name}:{line_number} is not valid JSON: {exc}"
                ) from exc
    return questions


def indexable_pages(doc: str) -> tuple[set[int], int]:
    """Pages of a document that survive parsing, and its total page count.

    A page that parse_pdf() drops is not in the index, so a gold label pointing
    at it is unreachable by any retriever. Parsing here - rather than trusting
    the page numbers - is the only way to know that.
    """
    pdf_path = RAW_DIR / doc
    if not pdf_path.exists():
        return set(), 0
    return {page.page_number for page in parse_pdf(pdf_path)}, page_count(pdf_path)


def main() -> int:
    questions = load_questions()
    problems: list[str] = []
    seen_ids: set[str] = set()

    # Parsing a 582-page PDF is slow, so each document is parsed at most once.
    page_cache: dict[str, tuple[set[int], int]] = {}

    for question in questions:
        qid = question.get("id", "<missing id>")

        if qid in seen_ids:
            problems.append(f"{qid}: duplicate id")
        seen_ids.add(qid)

        for field in ("category", "difficulty", "question", "expected_answer"):
            if not question.get(field):
                problems.append(f"{qid}: missing or empty field {field!r}")

        if question.get("category") not in CATEGORIES:
            problems.append(f"{qid}: unknown category {question.get('category')!r}")
        if question.get("difficulty") not in DIFFICULTIES:
            problems.append(f"{qid}: unknown difficulty {question.get('difficulty')!r}")

        refusal = question.get("expected_refusal")
        gold = question.get("gold") or []

        if refusal is not None:
            if refusal not in REFUSAL_KINDS:
                problems.append(f"{qid}: unknown expected_refusal {refusal!r}")
            if gold:
                problems.append(
                    f"{qid}: a refusal question must have no gold pages - "
                    "if the answer is in the corpus, it is not a refusal"
                )
            continue

        if not gold:
            problems.append(f"{qid}: no gold pages and not marked as a refusal")

        for entry in gold:
            doc = entry.get("doc", "")
            if doc not in page_cache:
                page_cache[doc] = indexable_pages(doc)
            available, total = page_cache[doc]

            if total == 0:
                problems.append(f"{qid}: document not found under data/raw/: {doc}")
                continue

            for page in entry.get("pages", []):
                if page > total:
                    problems.append(
                        f"{qid}: {doc} page {page} is past the end of the "
                        f"document ({total} pages)"
                    )
                elif page not in available:
                    problems.append(
                        f"{qid}: {doc} page {page} is dropped by parse.py "
                        f"(under MIN_PAGE_CHARS), so no retriever can ever "
                        "return it - relabel this question"
                    )

    by_category = Counter(q.get("category") for q in questions)
    by_difficulty = Counter(q.get("difficulty") for q in questions)
    gold_pages = sum(
        len(entry.get("pages", []))
        for q in questions
        for entry in (q.get("gold") or [])
    )

    print(f"{len(questions)} questions, {gold_pages} gold pages")
    print("  by category:   " + ", ".join(f"{k} {v}" for k, v in sorted(by_category.items())))
    print("  by difficulty: " + ", ".join(f"{k} {v}" for k, v in sorted(by_difficulty.items())))

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nAll questions validate against the corpus in data/raw/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

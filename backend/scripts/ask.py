"""
Ask a question from the terminal - the whole query pipeline, no web server.

Usage:

    python backend/scripts/ask.py "What did management say about attrition?"
    python backend/scripts/ask.py "Revenue in FY24?" --company tcs --year FY24

This exists so retrieval and answering can be debugged without the frontend in
the way. When an answer looks wrong, --show-chunks reveals whether the problem
is retrieval (the wrong passages came back) or generation (the right passages
came back and the model misread them). Those two failures have completely
different fixes, and the API alone cannot tell them apart.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.rag.answer import answer_question  # noqa: E402
from app.rag.store import search  # noqa: E402
from app.schemas import ChatRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the filings index.")
    parser.add_argument("question")
    parser.add_argument("--company", help="Filter, e.g. tcs")
    parser.add_argument("--year", help="Filter, e.g. FY24")
    parser.add_argument(
        "--doc-type",
        help="Filter: annual_report | quarterly_results | earnings_call",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print the retrieved passages before the answer.",
    )
    args = parser.parse_args()

    request = ChatRequest(
        question=args.question,
        company=args.company,
        fiscal_year=args.year,
        doc_type=args.doc_type,
    )

    # Checked before retrieval so the failure is one clear line, rather than a
    # GroqError traceback out of the client constructor. --show-chunks still
    # works without a key, which is useful for debugging retrieval alone.
    if not get_settings().groq_api_key and not args.show_chunks:
        print("GROQ_API_KEY is not set. Add it to .env (free key at "
              "https://console.groq.com/keys), or pass --show-chunks to "
              "inspect retrieval without generating an answer.")
        return 1

    chunks = search(request)

    if args.show_chunks:
        print(f"--- retrieved {len(chunks)} chunks ---")
        for index, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            print(f"\n[S{index}] {meta['company']} {meta['fiscal_year']} "
                  f"{meta['doc_type']} page {meta['page']}")
            print(chunk.page_content[:300].replace("\n", " ") + "...")
        print("\n--- answer ---")

    if not get_settings().groq_api_key:
        print("(no GROQ_API_KEY - stopping after retrieval)")
        return 0

    response = answer_question(request, chunks)
    print(response.answer)

    if response.citations:
        print("\nSources:")
        for citation in response.citations:
            print(f"  - {citation.company} {citation.fiscal_year} "
                  f"{citation.doc_type}, page {citation.page} "
                  f"({citation.source_path})")
    elif not response.refused:
        # An answer with no citations violates the project's core promise, so
        # say so loudly rather than letting it pass as a normal result.
        print("\nWARNING: answer contained no verifiable citations.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

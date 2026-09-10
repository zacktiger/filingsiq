"""
Build the search index: PDFs on disk -> vectors in Qdrant.

Usage (from the project root, with the virtualenv active):

    python backend/scripts/ingest.py
    python backend/scripts/ingest.py --recreate     # drop and rebuild

Run --recreate after changing the embedding model, chunk size, or chunking
strategy. Without it, new chunks are added alongside the old ones and retrieval
silently searches a mixture of both.
"""

import argparse
import sys
from pathlib import Path

# Make `app.*` importable when running this file directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.rag.chunk import chunk_document  # noqa: E402
from app.rag.loader import absolute_path, discover_documents  # noqa: E402
from app.rag.parse import page_count, parse_pdf  # noqa: E402
from app.rag.store import index_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index filings into Qdrant.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the existing collection before indexing.",
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"Embedding model : {settings.embedding_model} "
          f"({settings.embedding_spec.dim} dims)")
    print(f"Collection      : {settings.collection_name}")
    print(f"Chunking        : size={settings.chunk_size} "
          f"overlap={settings.chunk_overlap}")
    print()

    documents = discover_documents()
    if not documents:
        print("No PDFs found under data/raw/. See data/README.md for the layout.")
        return 1

    all_chunks = []
    for meta in documents:
        path = absolute_path(meta)
        pages = parse_pdf(path)
        chunks = chunk_document(meta, pages)
        all_chunks.extend(chunks)

        # Reporting pages-with-text against total pages is a cheap sanity check:
        # a scanned PDF with no text layer shows up here as "0 of 210 pages"
        # instead of quietly contributing nothing to the index.
        print(
            f"  {meta.source_path:<45} "
            f"{len(pages):>4} of {page_count(path):>4} pages -> "
            f"{len(chunks):>5} chunks"
        )

    print(f"\nEmbedding and indexing {len(all_chunks)} chunks "
          "(first run downloads the model, which takes a while)...")
    index_documents(all_chunks, recreate=args.recreate)
    print(f"Done. {len(all_chunks)} chunks in collection "
          f"{settings.collection_name!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

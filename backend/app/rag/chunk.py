"""
Stage 3 of the pipeline: pages -> chunks small enough to embed.

Why chunk at all: an embedding compresses a whole passage into one vector, so
the longer the passage, the more diluted its meaning. A 200-page annual report
as a single vector matches nothing well. Chunks are small enough to be
specific and large enough to stand alone as an answer.

Each page is split INDEPENDENTLY. Splitting page by page means no chunk ever
straddles a page boundary, so every chunk has exactly one correct page number
to cite. The cost is that a sentence spanning a page break gets cut - an
acceptable trade for citations that are always right.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.parse import ParsedPage
from app.schemas import DocumentMeta


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Create the text splitter, sized from config.

    "Recursive" means it tries a list of separators in order - paragraph breaks,
    then line breaks, then sentences, then words - and only resorts to cutting
    mid-word if nothing else fits. That tends to keep semantic units intact,
    which is why it is the usual default rather than a fixed-width slice.
    """
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Overlap repeats the tail of one chunk at the head of the next, so a
        # fact sitting near a boundary appears whole in at least one chunk.
        length_function=len,
    )


def chunk_document(meta: DocumentMeta, pages: list[ParsedPage]) -> list[Document]:
    """Turn one document's pages into embeddable LangChain Documents.

    The returned Documents carry the metadata that retrieval filters on and that
    citations are built from. Everything downstream - Qdrant payloads, the
    Citation objects the API returns - reads these same keys, so they are the
    contract between pipeline stages.
    """
    splitter = build_splitter()
    chunks: list[Document] = []

    for page in pages:
        for chunk_index, chunk_text in enumerate(splitter.split_text(page.text)):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        # --- provenance: what makes a citation verifiable ---
                        "company": meta.company,
                        "fiscal_year": meta.fiscal_year,
                        "doc_type": meta.doc_type,
                        "source_path": meta.source_path,
                        "page": page.page_number,
                        # --- extras from manifest.csv (may be None) ---
                        "company_name": meta.company_name,
                        "sector": meta.sector,
                        "source_url": meta.source_url,
                        # Position of this chunk within its page. Not used for
                        # retrieval; helpful when debugging why a specific
                        # passage was or was not returned.
                        "chunk_index": chunk_index,
                    },
                )
            )

    return chunks


# --- Phase 3: structure-aware chunking ---------------------------------------
#
# Fixed-size chunking is the honest baseline, not the intended end state. It
# cuts across section boundaries, so a chunk can begin mid-sentence under the
# wrong heading - and it discards document structure that filings have plenty
# of ("Management Discussion and Analysis", "Notes to Accounts").
#
# project.md schedules the upgrade for Phase 3 (structure-aware, parent-child
# chunks). The point of building the plain version first is that Phase 2's eval
# harness can then MEASURE whether the upgrade helps, instead of assuming it.

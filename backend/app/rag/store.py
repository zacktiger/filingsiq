"""
Stage 5 of the pipeline: store vectors in Qdrant and search them.

Qdrant (rather than an in-memory store) for three reasons, in increasing order
of importance:
  1. The index survives a restart. Re-embedding the corpus with bge-m3 on CPU is
     slow enough that losing it on every reload would be painful.
  2. It filters on payload fields, so "only Infosys, only FY24" is a real
     server-side filter and not a post-hoc slice of the results.
  3. It stores sparse vectors natively, which is what Phase 3's hybrid
     dense + BM25 search needs. Choosing it now avoids a migration later.

The collection is created by langchain-qdrant itself on first ingest. That is
deliberate: the library has its own convention for naming the vector inside the
collection, and hand-rolling the collection risks a mismatch that only shows up
as zero search results.
"""

import json
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from app.config import DATA_DIR, get_settings
from app.rag.embed import get_embeddings
from app.schemas import ChatRequest

# Build info written after each successful ingest. Not required by the app - it
# exists so you can answer "what is actually in this index?" without guessing,
# which matters once Phase 2 starts comparing eval runs.
INDEX_META_PATH = DATA_DIR / "index_meta.json"


class IndexMismatchError(Exception):
    """Raised when the existing collection does not match the configured model."""


def get_client() -> QdrantClient:
    """Connect to Qdrant."""
    return QdrantClient(url=get_settings().qdrant_url)


def _existing_vector_size(client: QdrantClient, collection: str) -> int | None:
    """Read the vector dimension of an existing collection, if any.

    Qdrant reports either a single unnamed vector config or a dict of named
    ones, depending on how the collection was created, so both shapes are
    handled here.
    """
    info = client.get_collection(collection)
    config = info.config.params.vectors

    if config is None:
        return None
    if isinstance(config, dict):
        # Named vectors: Phase 1 only ever creates one.
        first = next(iter(config.values()), None)
        return first.size if first else None
    return config.size


def assert_index_matches_model() -> None:
    """Fail loudly if the stored index was built with a different model.

    This is the guard that makes the embedding model a safe config setting.
    A vector's dimension is fixed by the model that produced it, so querying a
    1024-dim bge-m3 index with 384-dim bge-small vectors is not a degraded
    search - it is meaningless. Qdrant would reject it, but the error would be a
    dimension number with no explanation of what to do about it.
    """
    settings = get_settings()
    client = get_client()
    collection = settings.collection_name

    if not client.collection_exists(collection):
        raise IndexMismatchError(
            f"Qdrant collection {collection!r} does not exist. "
            "Build it first: python backend/scripts/ingest.py"
        )

    expected = settings.embedding_spec.dim
    actual = _existing_vector_size(client, collection)

    if actual is not None and actual != expected:
        raise IndexMismatchError(
            f"Collection {collection!r} stores {actual}-dimensional vectors, but "
            f"EMBEDDING_MODEL={settings.embedding_model} produces {expected}. "
            "The embedding model changed since this index was built. Re-index: "
            "python backend/scripts/ingest.py --recreate"
        )


def index_documents(chunks: list[Document], recreate: bool = False) -> None:
    """Embed chunks and upsert them into Qdrant.

    Creates the collection on first run. Pass recreate=True to drop and rebuild
    it - needed after changing the embedding model or the chunking strategy,
    since stale chunks would otherwise linger alongside the new ones.
    """
    settings = get_settings()
    client = get_client()
    collection = settings.collection_name

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)

    if client.collection_exists(collection):
        assert_index_matches_model()
        store = QdrantVectorStore.from_existing_collection(
            collection_name=collection,
            embedding=get_embeddings(),
            url=settings.qdrant_url,
        )
        store.add_documents(chunks)
    else:
        # from_documents creates the collection with the right vector config
        # and then adds the documents.
        QdrantVectorStore.from_documents(
            chunks,
            embedding=get_embeddings(),
            url=settings.qdrant_url,
            collection_name=collection,
        )

    _write_index_meta(len(chunks), recreate=recreate)


def _write_index_meta(chunk_count: int, recreate: bool) -> None:
    """Record what was just indexed, for reproducibility."""
    settings = get_settings()
    INDEX_META_PATH.write_text(
        json.dumps(
            {
                "built_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model": settings.embedding_model,
                "vector_dim": settings.embedding_spec.dim,
                "collection": settings.collection_name,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "chunks_added": chunk_count,
                "full_rebuild": recreate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def get_store() -> QdrantVectorStore:
    """Open the existing collection for querying."""
    settings = get_settings()
    assert_index_matches_model()
    return QdrantVectorStore.from_existing_collection(
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
        url=settings.qdrant_url,
    )


def build_filter(request: ChatRequest) -> qmodels.Filter | None:
    """Translate the request's optional filters into a Qdrant filter.

    Returns None when nothing was specified, which searches the whole corpus.
    Filtering happens inside Qdrant rather than after retrieval - otherwise
    asking for top 5 could return 5 chunks and then discard 4 of them for being
    the wrong company, leaving almost nothing to answer from.
    """
    conditions = [
        qmodels.FieldCondition(
            key=f"metadata.{field}", match=qmodels.MatchValue(value=value)
        )
        for field, value in (
            ("company", request.company),
            ("fiscal_year", request.fiscal_year),
            ("doc_type", request.doc_type),
        )
        if value
    ]
    return qmodels.Filter(must=conditions) if conditions else None


def search(request: ChatRequest) -> list[Document]:
    """Retrieve the chunks most similar to the question."""
    settings = get_settings()
    return get_store().similarity_search(
        request.question,
        k=settings.top_k,
        filter=build_filter(request),
    )

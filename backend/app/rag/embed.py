"""
Stage 4 of the pipeline: text -> vectors.

Embeddings run LOCALLY here, via sentence-transformers. That is not a shortcut,
it is a constraint worth stating plainly: Groq offers no embeddings endpoint, so
the hosted model answers questions but never produces the vectors.

Running embeddings locally also keeps the corpus re-indexable at zero marginal
cost, which matters because Phases 3 and 4 re-index repeatedly to measure
whether a retrieval change actually helped. A metered embeddings API would put
a price on exactly the experiment this project is built around.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


class PrefixedEmbeddings(Embeddings):
    """Wraps an embedding model to apply an asymmetric query prefix.

    Some BGE models are trained asymmetrically: a short search query is
    prefixed with an instruction, while indexed passages are not. Getting this
    wrong raises no error - retrieval just gets quietly worse - so the prefix is
    applied in one place, driven by the model registry in config.py.

    The prefix is applied ONLY in embed_query (the question), never in
    embed_documents (the filings). That asymmetry is the entire point.
    """

    def __init__(self, inner: Embeddings, query_prefix: str) -> None:
        self._inner = inner
        self._query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus passages - no prefix."""
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question - prefix applied if the model wants one."""
        return self._inner.embed_query(self._query_prefix + text)


@lru_cache
def get_embeddings() -> Embeddings:
    """Return the embedding model, loading it once per process.

    Caching is important, not just tidy: constructing HuggingFaceEmbeddings
    loads model weights into memory (bge-m3 is roughly 2GB). The FastAPI app
    calls this at startup so no request pays that cost.
    """
    settings = get_settings()
    spec = settings.embedding_spec

    inner = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        # normalize_embeddings=True scales every vector to unit length, which
        # makes a dot product equal cosine similarity. BGE models are trained
        # for cosine similarity, and the Qdrant collection is created with
        # Distance.COSINE to match - all three have to agree.
        encode_kwargs={"normalize_embeddings": True},
    )

    return PrefixedEmbeddings(inner, spec.query_prefix)

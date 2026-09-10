"""
Central configuration for FilingsIQ.

Every tunable value lives here, so no other module reads os.environ directly.
Values are loaded from environment variables or a local .env file (see
.env.example at the project root).
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# This file is at <root>/backend/app/config.py, so the project root is 3 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.csv"


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Facts about an embedding model that the rest of the pipeline depends on.

    These three values must stay together. A Qdrant collection is created with a
    FIXED vector size, so `dim` is not an independent setting you can tune - it
    is a property of the model. Storing them in one place means they cannot
    drift apart and produce a silently broken index.
    """

    dim: int
    query_prefix: str
    slug: str  # filesystem/collection-safe short name


# Registry of the embedding models this project supports.
#
# `query_prefix` exists because some BGE models are trained asymmetrically: a
# short search query gets an instruction prefix while the indexed passages do
# not. Using the wrong prefix does not raise an error - it just quietly returns
# worse results - which is exactly the kind of bug worth designing out.
#
#   bge-m3            - 1024 dims, multilingual, stronger, ~2.2GB download.
#                       Trained without a query instruction, so no prefix.
#   bge-small-en-v1.5 -  384 dims, far faster on CPU, English only.
#                       Its model card recommends a query instruction prefix.
EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "BAAI/bge-m3": EmbeddingModelSpec(
        dim=1024,
        query_prefix="",
        slug="bge_m3",
    ),
    "BAAI/bge-small-en-v1.5": EmbeddingModelSpec(
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        slug="bge_small_en_v15",
    ),
}


class Settings(BaseSettings):
    """Application settings, read from the environment / .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Answer generation (Groq free tier) ---
    groq_api_key: str = ""
    answer_model: str = "openai/gpt-oss-120b"

    # Temperature MUST stay at 0 for this workload. ChatGroq defaults to 0.7,
    # which is actively wrong here: the job is to copy figures out of a filing
    # verbatim, and sampling randomness is exactly how "Rs. 48,250 crore"
    # becomes "Rs. 48,520 crore" - a plausible-looking, unverifiable error.
    answer_temperature: float = 0.0

    # gpt-oss models reason before answering: low | medium | high.
    # Blank disables it. "medium" suits grounded Q&A, where the retrieved text
    # does the work and the model mainly has to read carefully and cite well.
    answer_effort: str = "medium"

    # Cap on answer length. Kept modest deliberately: the free tier allows
    # 200,000 tokens per DAY, and every output token is spent from the same
    # budget as the Phase 2 eval runs.
    answer_max_tokens: int = 1024

    # Client-side throttle, in requests per second. The free tier permits 30
    # requests/minute (0.5/sec); 0.4 leaves headroom so a burst of eval
    # questions queues locally instead of collecting HTTP 429s.
    requests_per_second: float = 0.4

    # --- Embeddings (run locally; no hosted provider offers them here) ---
    embedding_model: str = "BAAI/bge-m3"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"

    # --- Retrieval ---
    top_k: int = 5

    # Chunking. 1000/200 is a deliberately plain starting point; Phase 3 replaces
    # this with structure-aware chunking and measures whether it actually helps.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @property
    def embedding_spec(self) -> EmbeddingModelSpec:
        """Look up the dimension and query prefix for the configured model."""
        try:
            return EMBEDDING_MODELS[self.embedding_model]
        except KeyError:
            supported = ", ".join(EMBEDDING_MODELS)
            raise ValueError(
                f"Unknown EMBEDDING_MODEL {self.embedding_model!r}. "
                f"Supported models: {supported}. "
                "To add another, register its dimension and query prefix in "
                "EMBEDDING_MODELS in backend/app/config.py."
            ) from None

    @property
    def collection_name(self) -> str:
        """Qdrant collection name, derived from the embedding model.

        Deriving the name from the model (rather than hardcoding one) means
        switching models creates a NEW collection instead of writing vectors of
        the wrong size into the existing one. Swapping bge-m3 (1024 dims) for
        bge-small (384 dims) therefore requires a re-index - and this makes that
        requirement obvious instead of surprising.
        """
        return f"filings_{self.embedding_spec.slug}"


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton.

    Cached so the .env file is read once per process, and so FastAPI can use
    this as a dependency without re-parsing on every request.
    """
    return Settings()

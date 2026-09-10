"""
FastAPI application - the online query pipeline.

Phase 1 exposes the minimum that a frontend needs: ask a question, get an answer
with citations. The other endpoints in project.md (/search, /documents/{id},
/feedback, /ingest) and SSE streaming arrive in later phases.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.rag.answer import answer_question
from app.rag.embed import get_embeddings
from app.rag.store import IndexMismatchError, get_client, search
from app.schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model before the server accepts traffic.

    Without this, the first request pays the cost of loading ~2GB of model
    weights and appears to hang. Doing it at startup means a slow boot and fast
    requests, which is the right trade for a service.
    """
    settings = get_settings()
    print(f"Loading embedding model {settings.embedding_model}...")
    get_embeddings().embed_query("warmup")
    if not settings.groq_api_key:
        # A warning rather than a hard exit: retrieval and /health are still
        # useful for debugging the index without a key present.
        print(
            "WARNING: GROQ_API_KEY is not set. Retrieval will work, but "
            "/chat cannot generate answers. Add it to .env."
        )

    print(f"Ready. Collection: {settings.collection_name}")
    yield


app = FastAPI(
    title="FilingsIQ",
    description="Search and Q&A over Indian company filings, with citations.",
    version="0.1.0",
    lifespan=lifespan,
)

# The Vite dev server runs on a different port, so the browser treats API calls
# as cross-origin. Localhost only - a deployed frontend origin gets added in
# Phase 7 rather than loosening this to "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Report whether the pieces this service depends on are actually usable.

    Deliberately more than `{"status": "ok"}`: it reports whether Qdrant is
    reachable and whether the expected collection exists, because those are the
    two things that break and the failure is otherwise invisible until a query
    returns nothing.
    """
    settings = get_settings()
    result = {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "collection": settings.collection_name,
        "qdrant": "unreachable",
        "indexed_chunks": None,
        # Reported because a missing key is otherwise only discovered on the
        # first real question, as an opaque failure deep in the model call.
        "groq_key": "set" if settings.groq_api_key else "missing",
    }

    try:
        client = get_client()
        if client.collection_exists(settings.collection_name):
            info = client.get_collection(settings.collection_name)
            result["qdrant"] = "ok"
            result["indexed_chunks"] = info.points_count
        else:
            result["qdrant"] = "ok"
            result["status"] = "no_index"
    except Exception as exc:  # noqa: BLE001 - health must never itself 500
        result["status"] = "degraded"
        result["error"] = str(exc)

    return result


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question from the indexed filings."""
    if not get_settings().groq_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is not configured, so answers cannot be "
                "generated. Add it to .env and restart the server."
            ),
        )

    try:
        chunks = search(request)
    except IndexMismatchError as exc:
        # 503 rather than 500: the service is fine, the index is missing or was
        # built with a different embedding model. The message says how to fix it.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return answer_question(request, chunks)

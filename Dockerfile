# FilingsIQ - single-container deployment (Hugging Face Spaces, Docker SDK).
#
# One process serves the React UI and the API. The vector index ships inside
# the image (data/index_export, loaded into an in-memory Qdrant at startup), so
# there is no database to run. The only secret is GROQ_API_KEY, set in the
# Space's settings - never baked into the image.

# --- 1. Build the UI ---------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- 2. Python app -----------------------------------------------------------
FROM python:3.12-slim

# Spaces run the container as uid 1000; give it a writable home for model caches.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    QDRANT_LOCAL_INDEX=/app/data/index_export \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

WORKDIR /app

# CPU-only torch first: the default wheel bundles CUDA and is several GB larger,
# for a GPU a free Space does not have.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY data/index_export/ data/index_export/
COPY --from=frontend /frontend/dist frontend/dist

USER user
# Download the embedding model at build time, so a cold start does not wait on it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]

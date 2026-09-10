# FilingsIQ

Search and Q&A over Indian company filings — annual reports, quarterly results,
and earnings call transcripts — with **page-level citations** on every answer.

Ask "what did management say about attrition in FY24?" and get an answer that
names the document and the page, so the claim can be checked by hand.

> **Status: Phase 1 (vertical slice).** One end-to-end path works: PDF on disk →
> chunked → embedded → retrieved → cited answer, exposed over an HTTP API with a
> minimal React page. The full plan is in [`project.md`](project.md); what is and
> is not built yet is listed under [Roadmap](#roadmap).

## Documentation

| Document | What it covers |
|---|---|
| **[Thought Process](docs/THOUGHT_PROCESS.md)** | Why it is built this way — options rejected, measured trade-offs, and where the obvious choice was wrong |
| **[Architecture](docs/ARCHITECTURE.md)** | Component diagram, the provenance chain, data model, infrastructure |
| **[Technical Flow](docs/TECHNICAL_FLOW.md)** | Line-level walkthrough of indexing and query, with failure modes |
| [project.md](project.md) | The original spec: all 8 phases, dataset scope, evaluation plan |
| [CLAUDE.md](CLAUDE.md) | Constraints that silently break things if violated |

## What works today

- **Ingestion** — PDFs discovered from their path, text extracted page by page
- **Retrieval** — dense vector search over Qdrant, with metadata filters
  (company, fiscal year, document type)
- **Cited answers** — the model answers only from retrieved passages, and citations
  are resolved back against what was actually retrieved
- **Refusal** — declines when the filings do not contain the answer, and when
  asked to predict performance or give investment advice
- **Interfaces** — `POST /chat`, `GET /health`, a CLI, and a one-page React UI

## Architecture

A condensed view; the full component diagram and provenance chain are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
PDFs on disk
  │  loader.py    path → metadata (company / fiscal year / doc type)
  │  parse.py     PyMuPDF, one entry PER PAGE  ← page numbers captured here
  │  chunk.py     split each page independently ← page provenance preserved
  │  embed.py     bge-m3, running locally
  ↓  store.py     Qdrant (persistent, filterable, sparse-vector ready)
Qdrant
  ↓  store.py     dense search + metadata filter
  ↓  answer.py    gpt-oss-120b on Groq, grounded on retrieved passages only
answer + citations → FastAPI /chat → React
```

Two design points carry most of the weight:

**Page numbers are captured at parse time and never lost.** Pages are extracted
individually and each page is chunked on its own, so no chunk straddles a page
boundary and every chunk has exactly one correct page to cite. A sentence
spanning a page break gets cut — an acceptable price for citations that are
always right.

**Citations are resolved, not trusted.** The model marks sources as `[S1]`,
`[S2]`; those markers are then matched back against the chunks actually
retrieved. An invented `[S9]` has nothing to resolve to and is dropped.

## Setup

Requires **Python 3.10+**, **Node 18+**, and **Docker**.

```bash
# 1. Vector database
docker compose up -d
curl http://localhost:6333          # should return version info

# 2. Backend
python -m venv .venv
.venv/Scripts/activate              # Windows;  source .venv/bin/activate elsewhere
pip install -r backend/requirements.txt

# 3. Configuration
cp .env.example .env                # then add your GROQ_API_KEY

# 4. Frontend
cd frontend && npm install && cd ..
```

The first run downloads the bge-m3 embedding model (~2.2 GB). To skip that on a
slow connection, set `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` in `.env` — it is
384-dimensional instead of 1024, considerably faster on CPU, and English only.

### Add filings

Drop PDFs into the layout below and describe them in `data/manifest.csv` — see
[`data/README.md`](data/README.md).

```
data/raw/{company}/{fiscal_year}/{doc_type}.pdf
data/raw/tcs/FY24/annual_report.pdf
```

A synthetic filing (`data/raw/democo/FY24/annual_report.pdf`) is committed, so
the pipeline can be run end to end before downloading anything. Regenerate it
with `python backend/scripts/make_test_pdf.py`.

### Run

```bash
# Build the index
python backend/scripts/ingest.py

# Ask from the terminal (--show-chunks prints what retrieval returned)
python backend/scripts/ask.py "What did management say about attrition?"
python backend/scripts/ask.py "Revenue in FY24?" --company democo --show-chunks

# API + UI
uvicorn app.main:app --reload --app-dir backend    # http://localhost:8000/docs
cd frontend && npm run dev                          # http://localhost:5173
```

Re-run `ingest.py --recreate` after changing the embedding model, chunk size, or
chunking strategy — otherwise new chunks are added alongside the old ones and
retrieval searches a mixture of both.

## Notes for anyone extending this

**`temperature` must stay at 0.** `ChatGroq` defaults to **0.7**. This workload
copies figures out of filings verbatim, so sampling randomness is exactly how a
correct "Rs. 48,250 crore" becomes a plausible, unverifiable "Rs. 48,520 crore".
Nothing here wants variety. (Historical note: an earlier version of this project
targeted `claude-opus-5`, where the opposite rule applied — that model *rejects*
`temperature` with an HTTP 400. If you swap providers again, re-check which rule
applies.)

**The free tier's real limit is tokens per day, not requests.** 200,000 tokens/day
≈ 100 questions, and the Phase 2 eval draws on the same budget as interactive use.
`answer.py` installs a client-side `InMemoryRateLimiter` so a batch of eval
questions queues locally instead of collecting HTTP 429s part way through a run.

**Why raw PyMuPDF, not a LangChain document loader.** Every LangChain PDF loader
lives in `langchain_community.document_loaders` (being sunset), and the one
maintained standalone, `langchain-pymupdf4llm`, is AGPL-3.0. So `parse.py` calls
`pymupdf` directly. Measured against `PyMuPDFLoader` on the same files,
extraction is **byte-identical** — so this costs nothing in quality, and buys a
maintained dependency plus explicit control of the 0-based to 1-based page
conversion that every citation depends on. (`loader.py` has no loader
equivalent regardless: deriving metadata from the file path and merging
`manifest.csv` is not something a loader does.)

**`langchain-community` is deliberately absent.** It is being sunset, and most
tutorials for this stack import loaders and vector stores from it. Everything
here uses the maintained 1.x packages (`langchain-core`,
`langchain-text-splitters`, `langchain-qdrant`, `langchain-huggingface`,
`langchain-groq`).

**Embeddings are local and stay local.** Neither Groq nor Anthropic offers an
embeddings API, and a local model means re-indexing costs nothing — which matters
because Phases 3–4 re-index repeatedly to measure whether changes help. The
hosted model answers questions; it never produces vectors.

## Roadmap

Per [`project.md`](project.md). Phase 1 exists so the later phases can be
*measured* rather than assumed.

| Phase | Status |
|---|---|
| 1. Vertical slice | **done** |
| 2. Evaluation — 80–100 question test set, baseline metrics | next |
| 3. Retrieval upgrades — structure-aware chunking, hybrid search, reranking | |
| 4. Query understanding — intent, aliases, time parsing | |
| 5. Structured data — table extraction, text-to-SQL, LangGraph routing | |
| 6. Frontend — search mode, streaming, PDF viewer, charts | |
| 7. Production — deployment, tracing, caching, CI evals | |
| 8. Write-up — metrics table, architecture diagram | |

Phase 1 ships **no automated tests** on purpose. Phase 2 is the eval harness,
and its question set is the honest place to spend that effort — unit tests on
chunk boundaries would not tell us whether retrieval actually works.

### Known limitations

- **Tables are not extracted.** Flattening a financial table into a text chunk
  strips its row and column labels, and the model then misreports the numbers
  confidently. Phase 5 extracts tables into Postgres and answers numeric
  questions with SQL instead. Until then the prompt forbids arithmetic and the
  system reports only figures printed verbatim in the source.
- **Fixed-size chunking** cuts across section boundaries (Phase 3).
- **No hybrid search**, so exact-token queries (a specific metric name) are
  weaker than semantic ones (Phase 3).
- **Multi-column pages extract in scrambled reading order.** Measured: PyMuPDF
  interleaves two columns line by line, and `PyMuPDFLoader` produces exactly the
  same output — this is a property of PDF text extraction, not a bug in this
  code. Real filings are heavily multi-column, so this is the largest
  retrieval-quality risk here and the strongest argument for Phase 3's
  structure-aware chunking.
- **`MIN_PAGE_CHARS = 50` can discard a relevant page.** Measured: it drops a
  cover page that was the correct source for "which fiscal year does this report
  cover". Blank pages produce no chunks anyway (the splitter drops empty
  strings), so the threshold's only real effect is on short-but-meaningful
  pages. Worth revisiting once Phase 2 can measure it.
- **Answer throughput is capped by the free tier**, not by the code: ~100
  questions/day. Interactive use and eval runs share that budget.
- **Scanned PDFs with no text layer yield nothing.** `ingest.py` reports
  pages-with-text against total pages, so this shows up as "0 of 210 pages"
  rather than failing silently. No OCR.

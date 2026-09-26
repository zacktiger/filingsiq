---
title: FilingsIQ
emoji: 📑
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Cited Q&A over TCS, Infosys and HDFC Bank annual reports
---
how it's built: a React page on Vercel, calling a Gradio API on Hugging Face Spaces, with an in-memory Qdrant index and Groq for answers.
# FilingsIQ

Search and Q&A over Indian company filings — annual reports, quarterly results,
and earnings call transcripts — with **page-level citations** on every answer.

Ask "what did management say about attrition in FY24?" and get an answer that
names the document and the page, so the claim can be checked by hand.

> **Status: Phase 2 (evaluation), in progress.** Phase 1's end-to-end path works:
> PDF on disk → chunked → embedded → retrieved → cited answer, over an HTTP API
> with a minimal React page. Phase 2's question set and both halves of the eval
> — retrieval and answer quality — are built; the retrieval
> [baseline](#measured-retrieval-baseline) is measured and the answer baseline is
> the next run. The full plan is in [`project.md`](project.md); what is and is
> not built is listed under [Roadmap](#roadmap).

## Phase 1 corpus and verification

Indexed: the FY23 (2022-23) and FY24 (2023-24) integrated annual reports of
**TCS**, **Infosys**, and **HDFC Bank**, plus the synthetic DemoCo filing —
2,436 pages, 9,830 chunks with `bge-small-en-v1.5` (2,430 pages and 9,824 chunks
from the six real reports). Source URLs are recorded in
[`data/manifest.csv`](data/manifest.csv); the PDFs themselves are not committed.

Spot-checked by hand, when only the FY24 reports were indexed: each answer below
was compared against the text of the page it cites.

| Question | Answer | Cited page(s) | Verdict |
|---|---|---|---|
| TCS revenue in FY 2023-24? | ₹2,40,893 crore | 56, 76 | correct — the figure is on both pages |
| HDFC Bank net interest margin in FY24? | 3.53 per cent | 27, 216 | correct figure, but the source says *Core* NIM and the answer leaves out "Core" |
| TCS management on attrition? | trending down, credited to policies and learning | 75 | faithful, but leaves out the 12.5% figure printed on the same page |
| Infosys operating margin in fiscal 2024? | refused: "not in the filings" | — | **retrieval miss** — 20.7% is on page 16, which was not in the top 5 |
| Infosys revenue next year? | refused as a forecast | — | correct refusal |

The Infosys miss is the useful result here. The figure sits in a KPI tile ("Operating
margin 20.7%") with almost no surrounding prose, so a dense embedding of the
question does not rank it highly. An exact-token query like this is what Phase 3's
hybrid (BM25) search is meant to fix.

Phase 2 has since put a number on how often it happens — see below. It is
`lookup-19` in the question set, and it still misses.

## Measured retrieval baseline

Phase 2's question set is built: **121 questions**, each written by reading the
filings, with the expected answer and every page that states it
([`backend/eval/`](backend/eval/README.md)). Dense-only retrieval over 9,830
chunks (FY23 + FY24) scores:

| Category | n | Recall@5 | Recall@10 | Recall@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| **All** | **109** | **47.7%** | **57.8%** | **70.6%** | **0.380** |
| lookup | 72 | 51.4% | 61.1% | 72.2% | 0.411 |
| narrative | 11 | 45.5% | 72.7% | 81.8% | 0.317 |
| multi_year | 11 | 45.5% | 45.5% | 54.5% | 0.461 |
| arithmetic_boundary | 5 | 40.0% | 40.0% | 40.0% | 0.300 |
| cross_document | 10 | 30.0% | 40.0% | 80.0% | 0.176 |

```bash
python backend/eval/run_retrieval_eval.py    # reproduces the table; no tokens spent
```

Refusal questions are excluded — they have no gold page by definition, and are
scored by the answer eval instead. Every run is kept in `data/eval_runs/`.

**Adding a second year made retrieval worse, and that is the finding.** On the
FY24-only corpus the original 90 questions scored 51.1% Recall@5. Indexing the
FY23 reports dropped the same questions to 43.3% (41.1% before the gold lists
were extended to FY23 pages that state the same fact): in every one of the 9
lost hits, last year's report filled 3–5 of the top 5 slots, and in only one
did an FY23 page actually state the answer. Two consecutive annual reports are near-duplicates
in wording, so a dense embedding cannot tell "revenue in FY 2024" from
"revenue in FY 2023" — the year is one token in a long, otherwise identical
passage. 19 new FY23 questions show the same confusion from the other side:
39 of their 95 top-5 slots went to FY24 pages.

Measured with the correct company and year supplied as a filter (taken from
the gold labels, so an upper bound for Phase 4's time parsing):

| Filter | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|
| none (baseline) | 47.7% | 57.8% | 70.6% |
| company | 48.6% | 58.7% | 70.6% |
| company + fiscal year | 58.7% | 68.8% | 76.1% |

The company adds almost nothing — questions name it, and dense search already
stays inside the right company. The year is worth 11 points. The remaining 41%
of misses survive a perfect filter, so they are a retrieval-quality problem
that parsing the question cannot fix.

**Two results that reorder Phase 3.** Widening k from 5 to 20 buys only 23
points, so most misses are never retrieved rather than mis-ranked — and a
reranker can only reorder what dense search already returned. Hybrid BM25 and
structure-aware chunking come first; reranking second. The exception is
comparison questions, where recall climbs from 30% at k=5 to 80% at k=20 because one
company's filing can occupy every slot: that is a top-k or per-document-quota
fix, and it is nearly free.

On the FY24-only corpus, narrative questions (70%) beat factual lookups (50%) by
20 points — prose is what dense embeddings are good at, and exact metric names
are not. With FY23 indexed, the same 10 narrative questions fell to 50% — the
steepest drop of any category.

## Documentation

| Document | What it covers |
|---|---|
| **[Thought Process](docs/THOUGHT_PROCESS.md)** | Why it is built this way — options rejected, measured trade-offs, and where the obvious choice was wrong |
| **[Architecture](docs/ARCHITECTURE.md)** | Component diagram, the provenance chain, data model, infrastructure |
| **[Technical Flow](docs/TECHNICAL_FLOW.md)** | Line-level walkthrough of indexing, query and evaluation, with failure modes |
| **[Evaluation](backend/eval/README.md)** | The question set: schema, what each hard question breaks, scoring rules that are not obvious |
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
- **Evaluation** — 121 hand-labelled questions, a retrieval eval that runs
  without spending a token, and an answer eval that scores correctness,
  refusal and citations and says *which half* of RAG failed on each question

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

# Evaluation — these two call no model, so both are free to repeat
python backend/eval/validate_questions.py                  # labels still valid?
python backend/eval/run_retrieval_eval.py --label "my-change"

# Answer evaluation — spends ~1,750 tokens per question, ~1 day's budget in full
python backend/eval/run_answer_eval.py --label "my-change"
python backend/eval/run_answer_eval.py --resume data/eval_runs/answers-<stamp>.jsonl
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
≈ **114 questions** (measured on the real corpus: ~1,750 tokens per question,
input 1,325–1,805, at top_k=5). A full 121-question answer eval therefore needs
slightly more than one day's budget, shared with interactive use — which is why it is
resumable. An earlier figure of 965 tokens was measured on the small synthetic
fixture and understated the real cost by nearly half.

Two different rate limits apply. `answer.py`'s `InMemoryRateLimiter` caps
*requests* (0.4/s); the 8K tokens-per-*minute* limit is what a batch run hits
first, so the answer eval paces itself on tokens actually spent.

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
| 2. Evaluation — 80–100 question test set, baseline metrics | **in progress** — [121-question set](backend/eval/README.md), retrieval eval and answer eval built; retrieval baseline measured; answer baseline next |
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
- **Cited page numbers are PDF page indices, not printed page numbers.** The
  TCS and Infosys copies come from BSE, whose exchange filings put a
  cover letter in front of the report, so page 1 of those PDFs is the letter.
  Citations stay correct against the linked `source_url`, but they will not
  match the page numbers printed in the report.
- **Answer throughput is capped by the free tier**, not by the code: ~100
  questions/day. Interactive use and eval runs share that budget.
- **Scanned PDFs with no text layer yield nothing.** `ingest.py` reports
  pages-with-text against total pages, so this shows up as "0 of 210 pages"
  rather than failing silently. No OCR.

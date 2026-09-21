# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**FilingsIQ** — search and Q&A over Indian company filings (annual reports,
quarterly results, earnings call transcripts) with page-level citations.

`project.md` is the authoritative spec: 8 phases, dataset scope, evaluation
plan, success criteria. **Read it before proposing anything structural** — most
"missing" features are deliberately scheduled for a later phase, and the phase
ordering exists so retrieval changes can be *measured* rather than assumed.

**Currently at Phase 1 (vertical slice).** Working: ingestion → chunking →
dense retrieval → cited answer → `/chat` → React page.

## Commands

```bash
docker compose up -d                                  # Qdrant (required)

python backend/scripts/make_test_pdf.py               # regenerate synthetic test filing
python backend/scripts/ingest.py                      # build the index
python backend/scripts/ingest.py --recreate           # drop and rebuild
python backend/scripts/ask.py "question" --show-chunks

uvicorn app.main:app --reload --app-dir backend       # API on :8000
cd frontend && npm run dev                            # UI on :5173
```

`curl localhost:8000/health` reports Qdrant reachability and indexed chunk count
— check it first when queries return nothing.

## Non-obvious constraints

These will silently break things if violated.

1. **`temperature` must stay 0 — and `ChatGroq` defaults to 0.7.** This task
   copies figures out of filings verbatim, so sampling randomness produces
   plausible wrong numbers — the worst failure mode this project has.
   `config.py` pins `answer_temperature: float = 0.0`; do not "tune" it up.
   (An earlier revision targeted `claude-opus-5`, where the rule was *inverted*:
   that model rejects `temperature` with a 400. Re-check when changing provider.)

2. **Respect the free-tier token budget.** Groq free tier: 30 req/min,
   1,000 req/day, **200K tokens/day** — the last is binding (~207 questions,
   measured at 965 tokens each), and
   interactive use shares it with Phase 2 eval runs. `answer.py` installs an
   `InMemoryRateLimiter` at `REQUESTS_PER_SECOND=0.4`; removing it makes batch
   eval runs die on HTTP 429 partway through.

3. **Do not import from `langchain_community`.** It is being sunset. This project
   uses only maintained 1.x packages: `langchain-core`,
   `langchain-text-splitters`, `langchain-qdrant`, `langchain-huggingface`,
   `langchain-groq`. Tutorial code that imports
   `langchain_community.vectorstores.FAISS` does not belong here.

4. **Embedding dimension is fixed at collection-creation time.** bge-m3 is 1024,
   bge-small-en-v1.5 is 384. `Settings.collection_name` derives the collection
   from the model slug so switching models targets a *new* collection, and
   `assert_index_matches_model()` fails loudly otherwise. Changing
   `EMBEDDING_MODEL` requires `ingest.py --recreate`.

5. **Embeddings are local; no hosted provider is involved.** Groq offers no
   embeddings endpoint (nor does Anthropic). sentence-transformers runs them on
   CPU, so re-indexing is free — which Phases 3–4 depend on. Do not "simplify"
   this into a hosted embeddings call.

6. **Page numbers are the product.** `parse.py` extracts text per page and
   `chunk.py` splits each page independently, so every chunk has exactly one
   correct page to cite. Any change that concatenates pages before chunking
   destroys citation accuracy — the project's core promise. If you improve
   chunking (Phase 3), preserve one-page-per-chunk provenance or replace it with
   something demonstrably better.

7. **The query prefix is per model.** `bge-small-en-v1.5` needs a query
   instruction prefix; bge-m3 does not. Both live in `EMBEDDING_MODELS` in
   `config.py` alongside the dimension, so they cannot drift. Using the wrong
   prefix degrades retrieval without raising anything.

## Layout

```
project.md                  the spec — read first
backend/app/
  config.py                 settings + EMBEDDING_MODELS registry (dim, prefix, slug)
  schemas.py                DocumentMeta, ChatRequest, Citation, ChatResponse
  main.py                   FastAPI: POST /chat, GET /health
  rag/
    loader.py               data/raw path → metadata; fails loudly on bad layout
    parse.py                PyMuPDF, per page; pdfplumber tables = Phase 5 stub
    chunk.py                per-page splitting, carries provenance into metadata
    embed.py                local embeddings + PrefixedEmbeddings wrapper
    store.py                Qdrant: index, dim guard, filters, search
    answer.py               prompt, Groq call, rate limiter, citations, refusal
backend/scripts/            ingest.py, ask.py, make_test_pdf.py
backend/eval/               questions.jsonl (Phase 2 set), validate_questions.py
frontend/src/App.jsx        single-page UI (plain fetch + useState by design)
data/raw/{company}/{fy}/{doc_type}.pdf
data/manifest.csv           company_name, sector, source_url
```

## Conventions

- **Comments explain *why*, not what.** The existing comments justify design
  choices and name the trade-off not taken. Match that; don't restate the call
  on the next line.
- **Fail loudly on data problems.** A skipped filing is indistinguishable from a
  filing with no answer in it. `loader.py` raises `CorpusLayoutError` rather than
  skipping; `store.py` raises `IndexMismatchError` with the fix in the message.
- **Metadata keys are a contract.** `company`, `fiscal_year`, `doc_type`,
  `source_path`, `page` flow from `chunk.py` → Qdrant payload → `Citation`.
  Renaming one means updating all three and re-indexing.
- **Phase 1 has no automated tests, deliberately.** Phase 2 is the eval harness,
  and that is where the effort belongs. The question set now lives at
  `backend/eval/questions.jsonl` (102 questions, gold pages verified against
  parsed text); `validate_questions.py` guards it and costs no tokens. Don't add
  speculative unit tests on chunk boundaries — extend the eval instead.
- **Split the eval by cost.** Recall@k / MRR / nDCG need no model call, so they
  can run on every retrieval change; answer scoring costs ~965 tokens/question
  and ~half the daily budget per full run. Keep the two runners separate, or
  Phase 3 iterates twice a day instead of freely.
- Don't add dependencies from later phases (LangGraph, Postgres, Redis,
  Tailwind, TanStack Query) until that phase is actually being built.

## Gotchas

- `import pymupdf`, not `import fitz` — the `fitz` alias is deprecated in
  PyMuPDF 1.28 and warns on import.
- The scripts insert `backend/` on `sys.path` so `app.*` resolves when run
  directly; hence the `# noqa: E402` on their imports.
- `.env` is gitignored; `.env.example` is the template.
- Source PDFs are gitignored except `data/raw/democo/**`, the committed
  synthetic filing used to verify the pipeline without real downloads.

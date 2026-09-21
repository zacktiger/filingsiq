# Architecture

Component-level view of FilingsIQ at **Phase 1**. For the reasoning behind these
choices see [Thought Process](THOUGHT_PROCESS.md); for the step-by-step data path
see [Technical Flow](TECHNICAL_FLOW.md).

---

## System overview

Two pipelines that share only the vector store. Indexing runs offline and
occasionally; querying runs online per request.

```mermaid
graph TB
    subgraph offline["OFFLINE — indexing (scripts/ingest.py)"]
        direction TB
        PDF["data/raw/{company}/{fy}/{doc_type}.pdf"]
        L["loader.py<br/>path to metadata"]
        P["parse.py<br/>PyMuPDF, per page"]
        C["chunk.py<br/>split each page alone"]
        E1["embed.py<br/>bge-small, local CPU"]
        PDF --> L --> P --> C --> E1
    end

    QD[("Qdrant<br/>filings_bge_small_en_v15<br/>vectors + payload")]
    E1 -->|upsert| QD

    subgraph online["ONLINE — query (FastAPI)"]
        direction TB
        UI["React (Vite)<br/>question box"]
        API["main.py<br/>POST /chat"]
        E2["embed.py<br/>embed_query"]
        S["store.py<br/>dense search + filters"]
        A["answer.py<br/>gpt-oss-120b on Groq"]
        V["citation resolution<br/>+ refusal check"]
        UI --> API --> E2 --> S
        S --> A --> V
        V -->|answer + citations| UI
    end

    S -.->|similarity_search| QD
    QD -.->|top-k chunks| S

    style QD fill:#1f5fbf,color:#fff
    style A fill:#5a3fa8,color:#fff
    style V fill:#2d7d46,color:#fff
```

The dotted lines are the only coupling between the two pipelines. That is
deliberate: ingestion can be rewritten (Phase 3's structure-aware chunking)
without touching the query path, as long as the payload keys stay stable.

---

## Module responsibilities

Each module does one job and hands off. The `rag/` package has no knowledge of
HTTP, and `main.py` has no knowledge of PDFs.

| Module | Responsibility | Key decision it owns |
|---|---|---|
| `config.py` | Settings + embedding model registry | Ties dimension, query prefix and collection name to the model so they cannot drift |
| `schemas.py` | Pydantic contracts | `Citation` shape — what "verifiable" means concretely |
| `rag/loader.py` | Discover PDFs, derive metadata | Fails loudly on bad layout rather than skipping files |
| `rag/parse.py` | PDF to text, **per page** | Captures the page number that every citation depends on |
| `rag/chunk.py` | Split pages into embeddable units | Splits each page independently, preserving provenance |
| `rag/embed.py` | Text to vectors, locally | Applies the asymmetric query prefix in one place |
| `rag/store.py` | Qdrant index + search | Server-side metadata filtering; dimension guard |
| `rag/answer.py` | Prompt, model call, citations | Resolves `[Sn]` markers; detects refusal; rate limits |
| `main.py` | HTTP surface | Loads the model at startup, not per request |
| `eval/questions.jsonl` | 102 labelled questions | Gold is a *list* of pages — alternative correct sources, any one of which counts |
| `eval/validate_questions.py` | Guard the labels | Catches gold pages the parser drops, which would score 0 forever |
| `eval/run_retrieval_eval.py` | Recall@k, MRR, nDCG | Runs without a model call, so it is free to repeat |

---

## The provenance chain

The project's core promise — *every answer cites a document and page* — is a
chain, and it holds only if every link does. This is the most important diagram
in the repo.

```mermaid
graph LR
    A["PDF page 3"] -->|"parse.py<br/>page_number=3"| B["ParsedPage"]
    B -->|"chunk.py<br/>metadata['page']=3"| C["Document"]
    C -->|"store.py<br/>payload.metadata.page=3"| D["Qdrant point"]
    D -->|"similarity_search"| E["retrieved chunk"]
    E -->|"format_sources<br/>labelled [S1]"| F["prompt"]
    F -->|"model writes [S1]"| G["raw answer"]
    G -->|"build_citations<br/>resolves S1 to chunk"| H["Citation(page=3)"]
    H --> I["UI: 'page 3'"]

    style A fill:#8a6d1f,color:#fff
    style I fill:#2d7d46,color:#fff
```

Two properties worth noting:

- **The page number is never recomputed.** It is read once from the PDF and
  copied forward. Nothing downstream can derive a page, so nothing downstream
  can derive it *wrongly*.
- **The last arrow inverts the trust direction.** `build_citations` does not read
  the page from the model's text; it reads the marker, looks up the chunk that
  was actually retrieved, and takes the page from *that*. The model chooses
  *which* source to cite but never *what the citation says*.

---

## Data model

### On disk

```
data/raw/{company}/{fiscal_year}/{doc_type}.pdf     # path IS the metadata
data/manifest.csv                                    # what a path cannot express
```

`doc_type` is a closed set (`annual_report`, `quarterly_results`,
`earnings_call`) so a typo becomes an ingestion error rather than a phantom
document type no filter will ever match.

### In Qdrant

One point per chunk: a vector plus a payload. The payload keys are a **contract**
between `chunk.py`, the Qdrant filters, and `Citation`:

| Key | Role |
|---|---|
| `company`, `fiscal_year`, `doc_type` | Retrieval filters **and** citation display |
| `page` | Citation target (1-based, as a PDF reader shows it) |
| `source_path` | Locates the PDF; Phase 6 opens it at `page` |
| `company_name`, `sector`, `source_url` | From the manifest; display and traceability |
| `chunk_index` | Debugging only — position within its page |

Renaming any of the first four means updating `chunk.py`, `store.build_filter`,
`schemas.Citation`, **and re-indexing**.

---

## Infrastructure

```mermaid
graph LR
    B["Browser<br/>:5173"] -->|"/api proxy"| F["Vite dev server"]
    F --> A["FastAPI<br/>:8000<br/>+ bge-small in memory"]
    A --> Q["Qdrant<br/>:6333<br/>Docker + named volume"]
    A -->|HTTPS| G["Groq API<br/>gpt-oss-120b"]

    style G fill:#5a3fa8,color:#fff
    style Q fill:#1f5fbf,color:#fff
```

Only one process holds state: Qdrant, on a named Docker volume that survives
`docker compose down`. The API is stateless apart from the embedding model it
loads at startup — which is why startup is slow and requests are fast.

**A deployment consequence:** the embedding model lives *inside* the API process.
`bge-small` needs modest memory; `bge-m3` holds roughly 2.2 GB of weights. During
development the API process was killed by the OS under memory pressure while
running the *small* model, so a VPS sized for Phase 7 needs real headroom — or
embeddings need to move to their own service.

---

## The evaluation loop (Phase 2)

A third pipeline, and the reason it is drawn separately is that it splits at the
point where cost appears. Everything left of the dashed line is free and can run
on every change; everything right of it spends from a 200K tokens/day budget.

```mermaid
graph LR
    Q["questions.jsonl<br/>102 questions<br/>gold = doc + pages"]

    subgraph free["FREE — no model call"]
        direction TB
        V["validate_questions.py<br/>gold pages still parseable?"]
        R["run_retrieval_eval.py<br/>Recall@k · MRR · nDCG"]
    end

    subgraph paid["METERED — ~965 tokens/question"]
        direction TB
        AN["answer eval (not built)<br/>faithfulness · citation accuracy<br/>correct refusal rate"]
    end

    QD[("Qdrant")]
    PDFs["data/raw/*.pdf"]

    Q --> V
    Q --> R
    Q --> AN
    V -.->|re-parses| PDFs
    R -.->|similarity_search| QD
    AN -.->|POST /chat| QD
    R --> OUT["data/eval_runs/*.json<br/>one file per run"]

    style free fill:#1e3a2e,color:#fff
    style paid fill:#5a3fa8,color:#fff
    style QD fill:#1f5fbf,color:#fff
```

**Why the split is structural rather than a flag.** Retrieval metrics need only
the question and its gold pages, so they cost nothing and can run on every
chunking or embedding change — which is the loop Phase 3 lives in. Answer
metrics need a generation per question, so a full pass is roughly half a day's
token budget. Merged into one runner, measuring a chunking change would cost the
same as measuring answer quality, and Phase 3 would get about two experiments a
day.

**What `validate_questions.py` guards.** A gold page that `parse.py` discards
(under `MIN_PAGE_CHARS`) is not in the index, so no retriever can ever return
it. The question then scores 0 forever and reads as a retrieval failure rather
than a labelling mistake. The validator re-parses the PDFs and fails loudly on
exactly that — it is the same "fail loudly on data problems" convention as
`CorpusLayoutError` and `IndexMismatchError`, applied to the eval's own data.

**Why runs are written to `data/eval_runs/`, not just printed.** The success
criterion in [project.md](../project.md) is a baseline-versus-final metrics
table, which requires runs to be comparable months apart. Each file records the
embedding model, collection, chunk size and overlap alongside the scores,
because a recall change means nothing if the index underneath it also changed.
These files are committed — they are the evidence for every number in the
README. At ~108 KB per run they will need pruning eventually; the per-question
`retrieved` lists are the bulk and are what makes a regression diagnosable.

---

## What is deliberately absent

Phase 1 has no Postgres, Redis, Celery, LangGraph, reranker, BM25 index, or
streaming. Each is scheduled ([roadmap](../README.md#roadmap)) and each is
absent for the same reason: it would be an unmeasured addition to a system with
no baseline.

The one structural accommodation made for the future is Qdrant, chosen partly
because it stores sparse vectors natively — so Phase 3's hybrid retrieval is an
extension rather than a migration.

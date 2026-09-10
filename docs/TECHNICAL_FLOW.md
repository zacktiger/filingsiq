# Technical Flow

A line-level walkthrough of what happens to data, in order, with the actual
files and functions involved. Read [Architecture](ARCHITECTURE.md) first for the
component view, and [Thought Process](THOUGHT_PROCESS.md) for why any of it is
shaped this way.

---

## Part 1 — Indexing

Triggered by `python backend/scripts/ingest.py`. Runs offline; nothing here is
in the request path.

### 1.1 Discovery — `rag/loader.py`

`discover_documents()` walks `data/raw/**/*.pdf` (sorted, so log output is
stable between runs) and calls `parse_pdf_path()` on each.

Path parsing is strict by design:

```
data/raw/democo/FY24/annual_report.pdf
         └─company └─fy  └─doc_type
```

- Not exactly 3 path components → `CorpusLayoutError`
- `doc_type` not in `{annual_report, quarterly_results, earnings_call}` →
  `CorpusLayoutError`

Both raise rather than skip. **A silently skipped filing is indistinguishable
from a filing that contains no answer**, which is a genuinely painful bug to
chase later.

`load_manifest()` then reads `data/manifest.csv` keyed by `source_path` and
merges in `company_name`, `sector`, `source_url`. The manifest is optional — a
document without a row still indexes, just without those fields.

`source_path` is stored with forward slashes (`relative.as_posix()`) so the value
is byte-identical on Windows and Linux. It ends up in Qdrant payloads, so it must
not vary by machine.

**Out:** `list[DocumentMeta]`

### 1.2 Parsing — `rag/parse.py`

`parse_pdf()` opens the PDF with PyMuPDF and iterates pages:

1. `page.get_text("text")`
2. Normalise whitespace — strip each line, drop empty ones, rejoin with `\n`.
   PDF extraction emits runs of spaces and newlines from the original layout;
   left alone they waste chunk budget and carry no meaning.
3. Skip pages under `MIN_PAGE_CHARS = 50` — cover pages, blank separators,
   full-page images. They add retrieval noise and give the model nothing to cite.
4. Emit `ParsedPage(page_number=index+1, text=...)`

**The `+1` is the only place page numbering is converted.** PyMuPDF indexes from
0; readers display from 1. Converting once, here, is what stops off-by-one
citations pointing a reader at the wrong page.

`page_count()` reports total pages including skipped ones, so ingestion can print
`6 of 6 pages` — a cheap signal that extraction worked. A scanned PDF with no
text layer shows up as `0 of 210 pages` instead of silently contributing nothing.

**Out:** `list[ParsedPage]`

### 1.3 Chunking — `rag/chunk.py`

`chunk_document()` builds a `RecursiveCharacterTextSplitter`
(`chunk_size=1000`, `chunk_overlap=200`) and then — the important part — splits
**each page separately**:

```python
for page in pages:
    for chunk_index, chunk_text in enumerate(splitter.split_text(page.text)):
```

Because the splitter never sees two pages at once, no chunk can straddle a page
boundary, so every chunk has exactly one correct page to cite.

"Recursive" means the splitter tries separators in order — paragraph breaks,
line breaks, sentences, words — and only cuts mid-word if nothing else fits.

Each chunk becomes a LangChain `Document` carrying the payload contract:
`company`, `fiscal_year`, `doc_type`, `source_path`, `page`, plus manifest
extras and a debug-only `chunk_index`.

**Out:** `list[Document]`

### 1.4 Embedding — `rag/embed.py`

`get_embeddings()` is `@lru_cache`d, so weights load once per process rather than
once per call.

It wraps `HuggingFaceEmbeddings` in `PrefixedEmbeddings`, which exists for one
reason: some BGE models are trained **asymmetrically**. A short query gets an
instruction prefix; indexed passages do not.

```python
def embed_documents(self, texts):   # corpus — no prefix
    return self._inner.embed_documents(texts)

def embed_query(self, text):        # question — prefix applied
    return self._inner.embed_query(self._query_prefix + text)
```

The prefix comes from `EMBEDDING_MODELS` in `config.py`, alongside the dimension,
so the two cannot drift apart. Getting it wrong raises nothing — retrieval just
quietly degrades — which is why it is centralised rather than inlined.

`encode_kwargs={"normalize_embeddings": True}` scales vectors to unit length so a
dot product equals cosine similarity. This must agree with the collection's
`Distance.COSINE`; all three (model training, normalisation, distance metric)
have to match.

Measured on CPU: **21 ms/chunk** for `bge-small` (384-dim) vs **406 ms/chunk**
for `bge-m3` (1024-dim).

### 1.5 Upserting — `rag/store.py`

`index_documents()`:

1. If `--recreate`, drop the collection. Without it, new chunks are added
   *alongside* the old ones and retrieval searches a mixture of both.
2. If the collection exists → `assert_index_matches_model()`, then
   `QdrantVectorStore.from_existing_collection(...)` and `add_documents()`.
3. If not → `QdrantVectorStore.from_documents(...)`, which **creates the
   collection itself**.

Step 3 delegates creation deliberately. `langchain-qdrant` has its own
convention for naming the vector inside a collection; hand-rolling the
collection risks a mismatch whose only symptom is zero search results.

`_write_index_meta()` then records `data/index_meta.json` — model, dimension,
chunk size, count, timestamp. Not needed at runtime; it exists so "what is
actually in this index?" is answerable without guessing once Phase 2 starts
comparing eval runs.

**The dimension guard.** `assert_index_matches_model()` compares the
collection's stored vector size against the configured model's dimension and
raises `IndexMismatchError` naming the fix. Qdrant would reject a mismatch
anyway, but with a bare dimension number and no indication of what to do.

Because `Settings.collection_name` is derived from the model slug, changing
models targets a *different* collection rather than corrupting the current one —
so the two indexes coexist and the guard's usual job is reporting "this index
does not exist yet".

---

## Part 2 — Querying

### 2.1 Request — `main.py`

`POST /chat` with `ChatRequest`: `question`, plus optional `company`,
`fiscal_year`, `doc_type` filters.

Two guards run before any work:

- No `GROQ_API_KEY` → **503** with instructions. (Without this the failure was a
  raw `GroqError` traceback and an opaque 500.)
- `IndexMismatchError` from retrieval → **503**, not 500: the service is fine,
  the index is missing or stale, and the message says how to rebuild it.

The embedding model is loaded in the FastAPI `lifespan` handler, before the
server accepts traffic. Otherwise the first request pays the model-load cost and
appears to hang.

### 2.2 Filter construction — `store.build_filter()`

Supplied filters become Qdrant `FieldCondition`s on `metadata.<field>`, combined
with `must` (AND). No filters → `None` → search the whole corpus.

Filtering happens **inside Qdrant, not after retrieval**. Post-filtering is a
trap: ask for top-5, get 5 chunks, discard 4 for the wrong company, and answer
from 1. Server-side filtering means top-5 is five *relevant* chunks.

### 2.3 Retrieval — `store.search()`

```python
get_store().similarity_search(request.question, k=settings.top_k, filter=...)
```

`get_store()` re-asserts the dimension guard, then opens the existing collection.
The question is embedded through `PrefixedEmbeddings.embed_query`, so the query
prefix is applied here and only here.

**Out:** `list[Document]`, most similar first.

*Measured on the synthetic fixture: 5/5 top-1 correct, and a bogus company
filter correctly returns 0 hits.*

### 2.4 Prompt assembly — `answer.py::format_sources()`

Chunks are rendered as a numbered block:

```
[S1] DemoCo Limited | FY24 | annual_report | page 3
Management Discussion and Analysis
Attrition moderated significantly during the year...
```

The `[Sn]` marker is the handle the model cites and that resolution later
matches. The label gives the model enough to attribute precisely and gives a
reader enough to check.

The system prompt enforces six rules, of which three carry real weight:

- **Use only the SOURCES**, never outside knowledge about these companies.
- **No arithmetic across sources** — no growth rates, margins or CAGRs that are
  not printed. Numeric questions belong to Phase 5's SQL path; until then the
  system reports citable figures and says the calculation is unavailable.
- **Reply with a sentinel** (`INSUFFICIENT_CONTEXT` / `REFUSE_OUT_OF_SCOPE`)
  rather than apologising in prose.

### 2.5 Model call — `answer.py::get_model()`

`ChatGroq`, `@lru_cache`d, with three load-bearing settings:

| Setting | Value | Why |
|---|---|---|
| `temperature` | **0.0** | `ChatGroq` defaults to **0.7**. This task copies figures verbatim; randomness turns a correct `Rs. 48,250 crore` into a plausible, unverifiable `Rs. 48,520 crore`. |
| `rate_limiter` | 0.4 req/s | Free tier allows 30/min (0.5/s). Phase 2 fires 80–100 questions in a batch; throttling locally makes that queue instead of 429ing midway. |
| `max_tokens` | 1024 | The real ceiling is **200K tokens/day**, shared with eval runs. |

`reasoning_effort` is passed only when configured, so it can be disabled from
`.env` without touching code.

### 2.6 Refusal and citation resolution

`answer_question()`:

1. **No chunks retrieved** → return `refused=True` immediately, without spending
   a model call.
2. Invoke the model; normalise `.content` (a string for plain replies, a list of
   blocks when structured).
3. `INSUFFICIENT` or `OUT_OF_SCOPE` present → return the corresponding refusal
   with `refused=True` and no citations.
4. Otherwise resolve citations and return.

**Resolution is the guardrail.** `cited_indexes()` regex-matches `\[S(\d+)\]`
and **discards any marker outside `1..len(chunks)`**. `build_citations()` then
indexes into the retrieved chunks and reads `company`, `fiscal_year`, `page`
from *the chunk*, never from the model's text.

So a hallucinated `[S9]` against 5 sources has nothing to resolve to and
vanishes. The model chooses *which* source to cite; it never authors *what the
citation says*.

### 2.7 Response

```json
{
  "answer": "Voluntary attrition declined to 12.4 percent in FY24 [S1].",
  "citations": [
    {
      "company": "democo", "fiscal_year": "FY24",
      "doc_type": "annual_report", "page": 3,
      "source_path": "democo/FY24/annual_report.pdf",
      "quote": "Attrition moderated significantly during the year..."
    }
  ],
  "refused": false
}
```

`refused` is an explicit boolean, not something inferred from prose, so Phase 2
can score correct-refusal-rate without parsing text and the UI can style a
refusal differently from an error.

`quote` ships the retrieved chunk so a reader can verify without opening the PDF.

### 2.8 Rendering — `frontend/src/App.jsx`

Plain `fetch` and `useState`. The citation list is the point of the page. Two
deliberate behaviours:

- A **refusal** renders in muted italics under "Not answered" — declining is
  correct behaviour for a prediction question, not a failure.
- An answer with **zero citations** renders a warning telling the reader to treat
  it as unreliable, rather than displaying a bare paragraph. By this project's
  definition, an uncited answer is a failed answer.

---

## Failure modes and where they surface

| Failure | Where it is caught | What you see |
|---|---|---|
| Bad path under `data/raw/` | `loader.parse_pdf_path` | `CorpusLayoutError` naming the file and expected shape |
| Scanned PDF, no text layer | `ingest.py` reporting | `0 of 210 pages -> 0 chunks` |
| Index missing | `store.assert_index_matches_model` | 503 + "Build it first: ingest.py" |
| Embedding model changed | same guard | 503 + "Re-index: ingest.py --recreate" |
| `GROQ_API_KEY` unset | `main.chat`, `ask.py` | 503 / one-line CLI message |
| Qdrant down | `/health` | `{"status": "degraded", "qdrant": "unreachable"}` |
| Rate limit approached | `answer.get_model` limiter | Requests queue client-side |
| Model invents a citation | `answer.cited_indexes` | Marker discarded |
| Model answers uncited | CLI + UI | Explicit unreliability warning |

`GET /health` reports embedding model, collection name, Qdrant reachability,
indexed chunk count, and whether the Groq key is set — deliberately more than
`{"status": "ok"}`, because those are the things that actually break and their
failure is otherwise invisible until a query returns nothing.

---

## Debugging retrieval vs generation

When an answer is wrong, the first question is *which half failed*. That is what
`--show-chunks` is for:

```bash
python backend/scripts/ask.py "What about attrition?" --show-chunks
```

It prints the retrieved passages before the answer. If the right passages came
back and the answer is still wrong, that is a **generation** problem (prompt,
temperature, model). If the wrong passages came back, no prompt change will help
— that is a **retrieval** problem (chunking, embeddings, filters, top-k).

These two have entirely different fixes, and the API alone cannot tell them
apart. `--show-chunks` also works **without** an API key, so retrieval stays
inspectable when generation is unavailable.

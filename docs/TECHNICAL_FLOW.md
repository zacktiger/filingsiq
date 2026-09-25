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

## Part 3 — Evaluation

The retrieval eval reuses Part 2's search path exactly, stopping before the
model call. That is what makes it free, and it is also why its numbers describe
the real system: it calls the same `similarity_search`, against the same
collection, with the same embeddings.

### 3.1 Loading — `run_retrieval_eval.py::load_questions()`

Reads `questions.jsonl` and **drops the 12 refusal questions**. They carry no
gold pages by definition — the answer is not in the corpus — so there is nothing
for a retrieval metric to score. 90 questions remain.

**Out:** `list[dict]`, each with `question` and `gold`.

### 3.2 Gold expansion — `gold_pages()`

```python
{(entry["doc"], page) for entry in question["gold"] for page in entry["pages"]}
```

Flattens the per-document page lists into one set of `(doc, page)` pairs. These
are **alternatives, not a checklist**: TCS's FY24 revenue is stated on ten pages,
and retrieving any one of them is a correct result.

### 3.3 Search — `retrieved_pages()`

```python
hits = store.similarity_search(question, k=max_k)
[(h.metadata["source_path"].replace("\\", "/"), h.metadata["page"]) for h in hits]
```

One search per question at the largest cutoff, then sliced for each smaller `k` —
so `R@5`, `R@10` and `R@20` cost one query, not three.

The `replace("\\", "/")` is load-bearing. `source_path` is built from a Windows
path at ingest time, and the question set is written with POSIX separators.
Comparing them raw scores **every question as a miss** — a silent total failure
that looks like catastrophically bad retrieval rather than a bug.

### 3.4 Scoring — `reciprocal_rank()`, `ndcg_at_k()`, `documents_covered()`

| Metric | Question it answers |
|---|---|
| Recall@k | Was a correct page in the top k *at all*? Decides whether an answer is possible. |
| MRR@k | Was it near the top? Matters because the live setting is `top_k=5`. |
| nDCG@k | Were *several* gold pages found high up? Matters for multi-year and comparison questions. |
| documents covered | For `cross_document` only: is every company the question needs represented? |

The last one exists because Recall@k flatters comparisons. "Which of the three
spent most on CSR" counts as a hit the moment one company's page appears, though
it cannot be answered without all three. Reporting both makes the gap visible:
at k=5, recall is 40% but full document coverage is 50% — the documents are
often there while the specific right pages are not.

nDCG's ideal ranking is capped at `min(len(gold), k)`, so a question with two
gold pages is not penalised for failing to fill twenty slots it could never fill.

### 3.5 Output — `data/eval_runs/<timestamp>.json`

Each run records the scores **and** the index that produced them — embedding
model, collection, chunk size, overlap. A recall change means nothing if the
index underneath it also changed, and six weeks later nobody remembers which
settings a number came from.

**Out:** a printed table, plus a JSON file with per-question hits and the ranked
`retrieved` list for every question — which is what makes a regression
diagnosable rather than merely visible.

### 3.6 The guard — `validate_questions.py`

Run before an eval, and after any parsing or chunking change:

```bash
python backend/eval/validate_questions.py
```

It re-parses the PDFs and checks that every gold page **survives
`MIN_PAGE_CHARS`**. A gold page the parser drops is not in the index, so no
retriever can ever return it; the question scores 0 permanently and reads as a
retrieval failure. This is the eval's version of the project's "fail loudly on
data problems" convention.

It also checks that every **key fact** (`must_include`, `qualifiers`) appears in
the question's own `expected_answer`, using the same matcher the answer eval
uses. If someone edits the expected answer and forgets the key facts, the
validator fails instead of the eval silently scoring against a fact nobody
wrote down as correct.

## Part 4 — Answer evaluation

`run_answer_eval.py` runs Part 2 *in full*, including the model call — so unlike
Part 3 it spends tokens, about 1,750 per question.

### 4.1 Opening the run — `open_run()`

Creates `data/eval_runs/answers-<timestamp>.jsonl` plus a `.config.json` holding
the answer model, reasoning effort, temperature, `top_k`, embedding model and
chunk settings. With `--resume <file>` it instead reopens a partial run, reads
which question ids are done, and **refuses to continue if any setting changed**.

### 4.2 Warm-up

```python
get_embeddings().embed_query("warmup")
```

Loads the embedding model before the clock starts. Without it the first
question's latency includes ~50s of loading weights (measured). The LLM is
deliberately *not* warmed — that would spend tokens on nothing.

### 4.3 One question — `evaluate_one()`

```python
with get_usage_metadata_callback() as usage:
    chunks = search(request)
    response = answer_question(request, chunks)
```

Exactly the two calls `main.chat` makes. The `get_usage_metadata_callback`
context manager (from `langchain-core`) records the tokens every model call
inside it used, so the run's cost is measured, not estimated. For narrative
questions a second call — the faithfulness judge — happens inside the same
block, so its tokens are counted too.

### 4.4 Scoring — `score()` and `scoring.py`

| Check | Rule |
|---|---|
| correct | not refused, every `must_include` present, no `must_not_include` present |
| complete | correct, and every `qualifier` present |
| retrieval hit | every document the question needs had a gold page in the 5 retrieved chunks |
| diagnosis | retrieval hit × correct → one of four buckets |
| cited a gold page | any citation's `(doc, page)` is in gold |

`scoring.term_present()` compares numeric terms **as numbers**: it pulls every
number out of the answer, strips digit grouping, and checks membership. That is
what makes `2,40,893` equal `240,893` and stops `46` matching inside `146,463`.
Non-numeric terms are a case-insensitive substring match. `"b s r|bsr"` means
either spelling passes.

For refusal questions there is only one check: `correct = response.refused`.

### 4.5 The judge — `judge_faithfulness()`

Narrative questions only. The model receives the same numbered SOURCES block the
answer was written from (`format_sources`, reused from `answer.py`) and the
answer, and returns JSON listing each claim as supported or not. Faithfulness is
supported claims ÷ all claims. A reply that is not valid JSON is recorded as a
judge error rather than raised, so one bad reply cannot discard a run whose
earlier questions already cost tokens.

### 4.6 Pacing

```python
needed = tokens_this_question / tokens_per_minute * 60
time.sleep(max(0, needed - elapsed))
```

After each question the runner waits until that question's tokens fit inside a
7,000 tokens/minute budget, under Groq's 8,000. `answer.py`'s rate limiter
cannot do this — it counts requests, not tokens.

### 4.7 Stopping — and why an error is not a wrong answer

Any exception (most likely the daily token limit) **stops the run**. It is not
recorded as a wrong answer, because a quota error says nothing about answer
quality and would drag the baseline down for a reason unrelated to the system.
Everything answered so far is already on disk, and the runner prints the exact
`--resume` command.

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
| Gold page dropped by the parser | `validate_questions.py` | "page N is dropped by parse.py ... relabel this question" |
| Gold page past end of document | same | "page N is past the end of the document (M pages)" |
| Refusal question given gold pages | same | "a refusal question must have no gold pages" |
| Key fact drifted from expected answer | same | "key fact '3.53' does not appear in expected_answer" |
| Daily token limit mid-run | `run_answer_eval.py` | "Stopped at lookup-61 ..." plus the `--resume` command |
| Resuming after a settings change | `open_run()` | "Cannot resume: settings changed (top_k) ..." |
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

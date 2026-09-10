# Thought Process

Why FilingsIQ is built the way it is — including the options rejected and the
places where the obvious choice turned out to be wrong.

This document is about *reasoning*, not features. For what the system does see
the [README](../README.md); for how it fits together see
[Architecture](ARCHITECTURE.md) and [Technical Flow](TECHNICAL_FLOW.md).

---

## 1. The problem worth solving

A question like *"compare operating margins of three IT companies over three
years"* means opening nine PDFs of 200+ pages each and hunting through tables
whose row labels differ by company. Generic chatbots answer it instantly and
sometimes wrongly, with no way to check.

So the target was never "a chatbot for filings". It was: **an answer you can
verify in under ten seconds.** That single requirement drove most of what
follows — it is why page numbers are treated as a product feature rather than as
metadata, and why refusal is a first-class outcome instead of an error path.

## 2. Phase 1 is deliberately the boring version

The interesting parts of this project are hybrid search, reranking,
structure-aware chunking, and routing numeric questions to SQL. Phase 1 contains
none of them.

That is the point. Every one of those is an *optimisation*, and an optimisation
with no baseline is a guess. Building the plain version first — fixed-size
chunks, dense-only retrieval, no reranking — means Phase 2 can establish a
baseline and every later change has to earn its place against it.

**The risk I am accepting:** a reviewer skimming this repo at Phase 1 sees a
fairly ordinary RAG pipeline. I would rather defend a measured baseline than
demo an unmeasured stack of techniques.

## 3. Page numbers dictated the chunking strategy

The verifiability requirement has a concrete consequence that most RAG tutorials
get wrong: they concatenate a document into one string and then chunk it. Once
pages are joined, the page a sentence came from is unrecoverable.

So parsing emits `(page_number, text)` per page, and **each page is chunked
independently**. Every chunk therefore has exactly one correct page to cite.

**The cost:** a sentence spanning a page break gets cut in half, and neither
fragment reads well. I chose that over citations that are approximately right.
An approximate citation is worse than none — it invites a reader to trust a
number and then fail to find it.

## 4. Embeddings run locally, and the reason changed

Initially this was a hard constraint: the project first targeted Claude for
generation, and **Anthropic has no embeddings API**. After switching to Groq's
free tier the constraint held for the same reason — Groq has no embeddings
endpoint either.

But the better justification is economic. Phases 3 and 4 re-index the corpus
repeatedly to measure whether a retrieval change helped. A metered embeddings
API would put a per-experiment price on exactly the measurement this project is
built around. Local embeddings make re-indexing free, so there is never a reason
not to re-measure.

## 5. Choosing the embedding model: measured, not assumed

`bge-m3` is the stronger model and was the intended default. I benchmarked it
before committing, on this machine (CPU only — `torch 2.14.0+cpu`, no CUDA):

| Model | Dims | Cached load | Per chunk | 10k chunks |
|---|---|---|---|---|
| `BAAI/bge-m3` | 1024 | slow | **406 ms** | **~68 min** |
| `BAAI/bge-small-en-v1.5` | 384 | 6.5 s | **21 ms** | **~3.5 min** |

`bge-m3` also needs a one-time ~2.2 GB download (~15 minutes).

**19x slower.** For a one-off index that is tolerable; for the
iterate-and-measure loop of Phases 2–3 it is a deterrent to running the eval at
all. So `bge-small` is the Phase 1 default and `bge-m3` is reserved for the
final measured run.

The design consequence is more interesting than the choice. Vector dimension is
fixed when a collection is created, so switching models is a **re-index, not a
config flip**. Rather than leave that as a footgun, the collection name is
*derived* from the model (`filings_bge_small_en_v15` vs `filings_bge_m3`). Both
indexes coexist, switching is one env var plus one `--recreate`, and a mismatch
fails loudly with an actionable message instead of returning silent garbage.

## 6. Qdrant over FAISS or an in-memory store

The official LangChain RAG tutorial uses `InMemoryVectorStore`. Reasonable for a
tutorial, wrong here, for three reasons in increasing order of importance:

1. The index survives restarts — re-embedding is slow enough to matter.
2. Metadata filtering happens **server-side**. Filtering after retrieval is a
   trap: ask for the top 5, get 5 chunks, discard 4 for being the wrong company,
   and answer from one. Qdrant applies `company` / `fiscal_year` / `doc_type`
   *during* search, so top-5 means five *relevant* chunks.
3. Qdrant stores sparse vectors natively, which is what Phase 3's hybrid
   dense + BM25 retrieval needs. Choosing it now avoids a migration later.

Point 3 is what actually settled it. Points 1 and 2 have workarounds.

A note on cost: I pinned the Qdrant Docker image to an older version at first,
reasoning that a pinned version is more reproducible. It crashed on startup —
the modern `qdrant-client` that `langchain-qdrant` requires could not talk to
it. Pinning is right; pinning *without checking client compatibility* is not.

## 7. Two provider traps, pointing in opposite directions

The project first targeted `claude-opus-5`, then moved to Groq's free tier when
paid API access was ruled out. Both have a `temperature` trap, and they are
**inverses of each other**:

- **`claude-opus-5` rejects `temperature`** — sampling parameters were removed,
  and passing one returns HTTP 400. Since nearly every RAG tutorial sets
  `temperature=0`, tutorial-shaped code fails immediately.
- **`ChatGroq` defaults `temperature` to 0.7** — so tutorial-shaped code that
  *forgets* to set it silently gets a creative model. For a system whose job is
  copying figures out of a filing verbatim, that is how a correct
  `Rs. 48,250 crore` becomes a plausible, unverifiable `Rs. 48,520 crore`.

The second is far more dangerous, because it produces confident wrong numbers
instead of an error. It is pinned to `0.0` in config, and the reasoning is
recorded in [CLAUDE.md](../CLAUDE.md) so it does not get "tuned" back up.

**Generalisable lesson:** a wrong default that raises an error costs minutes; a
wrong default that changes output costs credibility. Audit the second kind.

## 8. Picking a free provider on evidence

With paid API access ruled out, I checked current free tiers rather than relying
on reputation:

| Provider | Free tier | Verdict |
|---|---|---|
| **Groq** (`openai/gpt-oss-120b`) | 30 req/min · 1,000 req/day · **200K tokens/day** | **Chosen** |
| Google Gemini | ~20 requests/**day** on most models | Rejected |
| Cerebras | $5 trial credits, not an ongoing tier | Rejected |

Gemini was the intuitive pick — 1M context, excellent tooling. Its free tier has
since been cut to roughly 20 requests/day, which **cannot run an 80–100 question
eval**. Since that eval is Phase 2's entire deliverable, the most attractive
option on paper was disqualified by the one number that mattered.

**The binding constraint is tokens/day, not requests/day.** At ~2K tokens per
question, 200K/day is about 100 questions — roughly *one full eval run per day*,
shared with interactive use. Two consequences:

- `answer.py` installs a client-side rate limiter (0.4 req/s against a 0.5
  limit) so a batch eval queues locally instead of collecting HTTP 429s halfway
  through a run.
- `max_tokens` is capped at 1024, because output tokens spend from the same
  budget as the eval.

This is a real limitation, not a solved problem. It caps iteration speed, and
the honest fix is paid access.

## 9. Refusal is a feature, and it must be machine-detectable

"Refuse when the answer isn't there" is easy to state and easy to fake. If the
model apologises in prose, then scoring *correct refusal rate* means
pattern-matching apologies — brittle, and it silently mis-scores.

So the model emits one of two sentinels — `INSUFFICIENT_CONTEXT` or
`REFUSE_OUT_OF_SCOPE` — which map to a boolean `refused` field on the response.
Phase 2 scores refusal exactly, and the frontend styles a refusal differently
from an error, because declining to answer a prediction question is *correct
behaviour*, not a failure.

## 10. Citations are resolved, never trusted

The model marks sources as `[S1]`, `[S2]`. Those markers are then matched back
against the chunks that were actually retrieved. A hallucinated `[S9]` against 5
retrieved sources has nothing to resolve to and is dropped.

This is the cheapest useful guardrail in the project: the model cannot invent
provenance, because provenance is *constructed from the retrieval result* rather
than parsed out of the model's prose.

An answer arriving with **zero** surviving citations is surfaced as a warning in
both the CLI and the UI rather than rendered as a normal paragraph — because by
this project's own definition, that answer is a failure.

## 11. Tables are excluded on purpose

Financial tables are the numeric heart of this corpus, and Phase 1 ignores them.

Flattening a table into a text chunk strips its row and column labels. The model
then reads a grid of loose numbers and reports them confidently and wrongly.
That is worse than not answering: it is an unverifiable error in the exact
category — numbers — where this project claims to be trustworthy.

So the interim position is stated rather than hidden: the prompt **forbids
arithmetic** and permits only figures printed verbatim in the retrieved text.
Phase 5 extracts tables into Postgres and routes numeric questions through SQL,
so the LLM never computes.

## 12. What I would challenge if I were reviewing this

- **Dense-only retrieval is the weakest link.** Exact-token queries (a specific
  metric name) are precisely where embeddings underperform BM25, and financial
  questions are full of them. Phase 3 addresses it, and I expect the measured
  gain to be large.
- **Fixed 1000/200 chunking is unjustified.** Those numbers are conventional,
  not derived. Phase 2 exists partly to find out whether they are any good.
- **`bge-small-en-v1.5` is English-only.** Indian filings contain some
  non-English content. `bge-m3` is multilingual, which is a second argument for
  it beyond raw quality.
- **No automated tests in Phase 1**, deliberately — unit tests on chunk
  boundaries would pass while retrieval quietly returned the wrong passages. The
  eval set is the test that matters and it is Phase 2's deliverable. But until
  then this repo's correctness rests on manual checks, which is a real gap.
- **One synthetic fixture is not a corpus.** Retrieval scores 5/5 top-1 on a
  6-page generated document. That verifies the *plumbing* and says nothing about
  quality on a 300-page annual report.

## 13. A process note

While building the fixture generator I wrapped text at a fixed character width,
which split words mid-token (`twelv` / `e month`). Extraction then produced
broken tokens — making the fixture *misrepresent the parser*, showing retrieval
as worse on test data than it would be on real PDFs, which wrap properly.

Worth recording for two reasons. First, test data has bugs too, and a fixture
that lies about the system is worse than no fixture. Second, my initial fix
silently failed — a string replacement that matched nothing — and my verification
check was wrong in the same direction, so I briefly believed it was fixed when it
was not. The lesson: **assert that an edit actually applied**, and write checks
that can fail loudly rather than quietly agree with you.

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

**The binding constraint is tokens/day, not requests/day.** Measured on this
pipeline: **965 tokens per question** (810 in, 155 out, at `top_k=5`), ~4.7s
latency. That is ~207 questions/day — about *two* full eval runs, shared with
interactive use. (My pre-measurement estimate was ~2K tokens and ~100
questions; the real figure is twice as generous.) Two consequences:

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

## 12. The question set had to come before the runner

The obvious order is to build the eval harness and then feed it questions. I did
the opposite, because the harness is the easy half — the scoring maths is thirty
lines — and the question set is where an eval is actually won or lost.

The temptation with a RAG eval is to generate questions *from the system*: ask
it things, keep what it answers well, label those as the test set. That set then
measures nothing, because it was selected for being answerable. So every
question here was written by reading the filings: I dumped all 1,272 pages of
text, hunted for facts worth asking about, and recorded the answer and the page
before ever running a query.

That produced **102 questions**, and it produces a specific kind of question the
other method never would — `lookup-52` asks how many contract employees HDFC
Bank has, which is one sentence on one page of 582, mentioned nowhere else. A
system-generated set contains no such question by construction.

**The trade-off I accepted:** this is slow, and it is limited by my own reading.
I covered the FY24 annual reports of three companies well; I could not have done
this for fifteen companies across three years, which is what the v1 scope in
[project.md](../project.md) eventually asks for. Scaling the set will need
either much more time or a labelling assistant — and if it is the latter, the
selection bias above comes back and has to be designed against.

## 13. The eval splits in two, and the reason is the token budget

The single most consequential decision in Phase 2 is that there are **two
runners, not one**.

Retrieval metrics — Recall@k, MRR, nDCG — need the question and its gold pages
and nothing else. No model call. No tokens. Answer metrics — faithfulness,
citation accuracy, correct refusal — need a generation per question, at ~965
tokens each, so one full pass over 102 questions is about half of Groq's
200K/day.

If those live in one script, then measuring whether a chunking change helped
costs half a day's budget, and Phase 3 gets roughly two experiments per day.
Split, the retrieval half is free and unlimited, and the expensive half runs
only when comparing final answer quality.

This is the same reasoning as §4 on local embeddings, arriving from a different
direction: **the parts of the loop I want to run constantly must not be metered.**
Local embeddings make re-indexing free; splitting the eval makes re-measuring
free. Together they mean a Phase 3 experiment costs time and nothing else.

## 14. The baseline, and the labelling bug that nearly hid it

Dense-only retrieval over 5,095 chunks scores **Recall@5 = 48.9%** across the 90
non-refusal questions (`backend/eval/run_retrieval_eval.py`, full table in the
[README](../README.md#measured-retrieval-baseline)). Under half the questions
retrieve a correct page at the live `top_k=5`.

Two readings change what Phase 3 should do first.

**Widening k from 5 to 20 buys only 21 points** (48.9% to 70.0%). So most misses
are not mis-ranked — they are never retrieved. A cross-encoder reranker can only
reorder what dense search already returned, which means reranking cannot fix the
bulk of this. Hybrid BM25 and better chunking come first. I had expected the
opposite going in, and would have spent Phase 3 on the reranker.

**Comparison questions behave differently.** `cross_document` recall goes 30% at
k=5 to 70% at k=10, and all required documents are present in the top 20 for
100% of them. Those chunks are being found and then crowded out, because one
company's filing can occupy every slot. That is a top-k or per-document-quota
fix, and it is nearly free — worth doing before anything expensive.

**The bug worth recording:** my first run scored `lookup-01` — "what was TCS's
revenue in FY 2024" — as a miss, even though the README already documented that
exact question working. The system was right and my labels were wrong: I had
labelled 3 pages, and the figure is actually stated on 10, including the
consolidated P&L that retrieval was correctly returning.

**An incomplete gold list reports a correct retrieval as a failure.** I fixed it
by searching the corpus for each answer's distinctive figure and keeping pages
where a matching label sits within 400 characters of it, which grew the set from
189 to 281 gold pages. It is still not complete: facts identified only by a
percentage (`20.7%`, `3.53 per cent`) are too common as tokens to expand that
way, so a few questions' recall is understated. Stated here because a baseline
whose error direction is unknown is not much of a baseline.

## 15. What I would challenge if I were reviewing this

- **Dense-only retrieval is the weakest link — now measured, not assumed.**
  Exact-token queries (a specific metric name) are precisely where embeddings
  underperform BM25, and financial questions are full of them. Recall@5 is
  48.9%, and narrative questions (70%) outscore factual lookups (50%) by 20
  points — which is the case for hybrid search stated as one number, since
  narrative prose is exactly what dense embeddings are good at.
- **Fixed 1000/200 chunking is unjustified.** Those numbers are conventional,
  not derived. Phase 2 exists partly to find out whether they are any good.
- **`bge-small-en-v1.5` is English-only.** Indian filings contain some
  non-English content. `bge-m3` is multilingual, which is a second argument for
  it beyond raw quality.
- **There are still no unit tests**, deliberately — tests on chunk boundaries
  would pass while retrieval quietly returned the wrong passages. The eval set
  is the test that matters, and the retrieval half of it now runs on every
  change for free. The answer half is not built yet, so faithfulness and
  citation accuracy still rest on hand-checked examples.
- **One synthetic fixture is not a corpus.** Retrieval scores 5/5 top-1 on a
  6-page generated document. That verifies the *plumbing* and says nothing about
  quality on a 300-page annual report.

## 16. A process note

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

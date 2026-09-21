# Phase 2 — evaluation question set

102 questions with hand-verified answers and gold page labels, written by
reading the indexed filings rather than by asking the system what it could
answer. Every gold page was checked against the text `parse.py` actually
extracts, so a label can never point at a page the pipeline discards.

```bash
python backend/eval/validate_questions.py     # no model, no tokens, run it freely
```

## Why the set is split from the runner

Retrieval metrics — Recall@k, MRR, nDCG — need only the question and its gold
pages. **No LLM call, no Groq token.** That half can run on every chunking or
embedding change, which is the loop Phase 3 lives in.

Answer metrics — faithfulness, correctness, citation accuracy, refusal rate —
cost roughly 965 tokens per question. 102 questions is about half the free
tier's 200K tokens/day, so the answer half is a deliberate, occasional run and
the retrieval half is not.

## Schema

One JSON object per line in `questions.jsonl`:

| Field | Meaning |
|---|---|
| `id` | `category-NN`, stable — results are joined on it across runs |
| `category` | `lookup`, `narrative`, `multi_year`, `cross_document`, `arithmetic_boundary`, `refusal` |
| `difficulty` | `easy` / `medium` / `hard`, assigned from *retrieval* difficulty, not reading difficulty |
| `question` | asked verbatim, with no filters set — filters are Phase 4 |
| `expected_answer` | the fact as printed in the filing |
| `gold` | `[{doc, pages}]` — every page where the fact is stated |
| `expected_refusal` | `null`, or `insufficient_context` / `out_of_scope` |
| `note` | why the question is here and what its failure mode looks like |

`gold` lists **all** pages carrying the fact, so Recall@k counts a hit if any
one of them is retrieved. Pages are PDF indices as `Citation.page` reports
them — the BSE copies of the TCS and Infosys reports carry a cover letter, so
these do not match the page numbers printed inside the reports.

## Distribution

| Category | N | What it measures |
|---|---|---|
| `lookup` | 54 | single fact, one document |
| `narrative` | 10 | management commentary, qualitative |
| `multi_year` | 11 | figures across years, from the history tables inside the FY24 reports |
| `cross_document` | 10 | two or three companies in one answer |
| `arithmetic_boundary` | 5 | must report figures and decline the calculation |
| `refusal` | 12 | 6 prediction/advice, 6 out-of-corpus |

26 are marked `hard`. They are not padding — each one names a specific
mechanism it breaks:

- **KPI tiles with no prose** (`lookup-19`, `lookup-20`). Infosys's 20.7%
  operating margin sits in a tile with nothing around it. This is the failure
  already recorded in the root README, and the clearest single measure of
  whether Phase 3's hybrid search earns its place.
- **Acronyms inside flattened tables** (`lookup-28` DSO, `lookup-38` CASA).
  Exact-token retrieval should win these and dense retrieval should lose them.
- **Deep table lookups** (`lookup-17` FY2019 revenue, `lookup-29` effective tax
  rate). Values far from their column headers once the table is flattened.
- **Needles** (`lookup-52`: 38 contract employees, one sentence in 582 pages).
- **Qualifier drops** (`lookup-35`: the source says *Core* NIM, not NIM).
- **Lexical-match refusals** (`refusal-07`, `refusal-09`, `refusal-12`). "Wipro"
  and "ICICI Bank Limited" both appear in the Infosys report as former
  employers and client names, so retrieval returns a confident-looking chunk
  for a question the corpus cannot answer.

## Scoring rules that are not obvious

**`arithmetic_boundary` questions are correct when the system refuses to
calculate.** Prompt rule 4 forbids arithmetic across sources, because Phase 5
routes numeric questions to SQL instead. `arith-05` asks for a subtraction
whose inputs sit in a single sentence; answering "20 basis points" is a
violation even though it is right. Scoring it as correct would hide the exact
behaviour Phase 5 exists to replace.

**`arith-03` inverts that.** It demands a calculation whose result is already
printed (4.7%). Refusing is over-refusal; computing it is a rule violation;
citing the printed figure is the only correct answer.

**`narrative-08` must not be refused.** It asks for HDFC Bank's economic
outlook, which is forward-looking but quoted from the filing. It is the control
against over-refusal on the boundary with `refusal-03`.

**Partial credit is worth recording separately** for questions whose note names
a qualifier (`lookup-02`, `lookup-22`, `lookup-24`, `lookup-35`). "3.53 per
cent" and "Core NIM 3.53 per cent" are different answers.

## Measured retrieval baseline

```bash
python backend/eval/run_retrieval_eval.py --label "phase1-baseline-dense-only"
```

Dense search only, `bge-small-en-v1.5`, 5,095 chunks, no filters, 90 non-refusal
questions. No model call — this run cost nothing and can be repeated freely.
Raw results: `data/eval_runs/`.

| Category | n | Recall@5 | Recall@10 | Recall@20 | MRR@20 | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| **All** | **90** | **48.9%** | **58.9%** | **70.0%** | **0.388** | **0.367** |
| narrative | 10 | 70.0% | 70.0% | 80.0% | 0.386 | 0.652 |
| multi_year | 11 | 54.5% | 63.6% | 81.8% | 0.497 | 0.484 |
| lookup | 54 | 50.0% | 57.4% | 70.4% | 0.401 | 0.338 |
| cross_document | 10 | 30.0% | 70.0% | 70.0% | 0.297 | 0.192 |
| arithmetic_boundary | 5 | 20.0% | 20.0% | 20.0% | 0.200 | 0.200 |

`cross_document`, all required documents present in top k: **k=5 50%, k=10 90%,
k=20 100%**.

Two readings worth acting on:

**Widening k from 5 to 20 buys only 21 points overall.** Most misses are not
mis-ranked, they are not retrieved at all. A cross-encoder reranker can only
reorder what dense search already returned, so on this evidence reranking is
the *second* Phase 3 lever, not the first — hybrid BM25 and structure-aware
chunking come first.

**`cross_document` is the exception: 30% at k=5, 70% at k=10.** For comparison
questions the right chunks *are* being found and then crowded out, because one
company's document can occupy all five slots. That is a top-k or per-document
quota problem, not a retrieval-quality problem, and it is cheap to fix.

Narrative questions retrieve best (70%), which is what dense embeddings are
good at. The gap between narrative and lookup is the whole argument for hybrid
search in one number.

## Two things the metric above does not yet capture

**Cross-document recall is scored leniently.** A question counts as a hit the
moment *any* gold page is retrieved, but `cross-05` needs all three companies
present to be answerable at all. The runner reports both — and the gap is
instructive: at k=5, recall is 30% while full document coverage is 50%, meaning
the right *documents* often reach the prompt while the right *pages* do not.
Neither number alone tells you that.

**Gold lists are exhaustive only where the fact has a distinctive figure.**
Labels were expanded by searching the corpus for each answer's figure and
keeping pages where a matching label sits within 400 characters of it — which
grew the set from 189 to 281 gold pages and turned `lookup-01` from a false
miss into a hit. Facts identified only by a percentage (`20.7%`, `3.53 per
cent`) could not be expanded that way, so a few gold lists may still be
incomplete and those questions' recall is understated rather than overstated.

## Corpus gaps this set cannot cover

The corpus is FY24 annual reports for three companies. Two consequences:

- **No transcript questions.** project.md's `transcript_search` category needs
  earnings call transcripts, which are not ingested. The `narrative` category
  stands in — management commentary from MD&A and the CEO letter has the same
  retrieval shape (qualitative, no figures) — but it is not the same test.
  `refusal-10` and `refusal-11` currently expect refusal *because* the
  documents are absent; they become answerable questions once transcripts are
  ingested, so they double as a corpus coverage tracker.
- **`multi_year` questions are answered from the history tables inside the FY24
  reports**, not from separate FY22/FY23 filings. Both TCS (5-year charts on
  p18, 10-year table on p90) and Infosys (5-year table on p16) publish enough
  history for real multi-year questions. This tests retrieval over dense tables,
  which is worth measuring on its own, but it does not test retrieval *across*
  documents for the same company.

## A parsing finding worth carrying into Phase 3

The rupee sign does not survive extraction consistently. In the HDFC Bank
report it comes out as `H` (`H85.8`, `H60,812.3 crore`) on some pages and `L`
or a backtick on others; the Infosys report uses a backtick (`` `577 cr ``).

This matters for Phase 3 specifically: a BM25 index will not match a query
token containing a currency symbol against these pages, so hybrid search will
look weaker than it is unless the currency glyph is normalised at parse time.
Six questions in this set (`lookup-27`, `lookup-41`, `lookup-48`, `lookup-50`,
and the CSR comparisons) sit directly on that fault line.

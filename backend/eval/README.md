# Phase 2 — evaluation question set

121 questions with hand-verified answers and gold page labels, written by
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

Answer metrics — correctness, citation accuracy, refusal rate, faithfulness —
cost roughly 1,750 tokens per question on the real corpus. 121 questions is
slightly more than the free tier's 200K tokens/day, so the answer half is a
deliberate, occasional run and the retrieval half is not.

## Schema

One JSON object per line in `questions.jsonl`:

| Field | Meaning |
|---|---|
| `id` | `category-NN`, stable — results are joined on it across runs |
| `category` | `lookup`, `narrative`, `multi_year`, `cross_document`, `arithmetic_boundary`, `refusal` |
| `difficulty` | `easy` / `medium` / `hard`, assigned from *retrieval* difficulty, not reading difficulty |
| `question` | asked verbatim, with no filters set — filters are Phase 4 |
| `expected_answer` | the fact as printed in the filing |
| `must_include` | terms a correct answer must contain — all of them, or it is **wrong** |
| `qualifiers` | terms a *complete* answer contains — missing one makes it incomplete, not wrong |
| `must_not_include` | terms that mean a rule was broken, e.g. a computed difference |
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
| `lookup` | 72 | single fact, one document — 54 about FY24, 18 about FY23 (`lookup-55`–`lookup-72`) |
| `narrative` | 11 | management commentary, qualitative |
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
python backend/eval/run_retrieval_eval.py --label "phase2-baseline-fy23-fy24-gold-v3"
```

Dense search only, `bge-small-en-v1.5`, 9,830 chunks (FY23 + FY24 reports), no
filters, 109 non-refusal questions. No model call — this run cost nothing and
can be repeated freely. Raw results: `data/eval_runs/20260925T124838Z.json`.

| Category | n | Recall@5 | Recall@10 | Recall@20 | MRR@20 | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| **All** | **109** | **47.7%** | **57.8%** | **70.6%** | **0.380** | **0.308** |
| lookup | 72 | 51.4% | 61.1% | 72.2% | 0.411 | 0.302 |
| narrative | 11 | 45.5% | 72.7% | 81.8% | 0.317 | 0.450 |
| multi_year | 11 | 45.5% | 45.5% | 54.5% | 0.461 | 0.382 |
| arithmetic_boundary | 5 | 40.0% | 40.0% | 40.0% | 0.300 | 0.249 |
| cross_document | 10 | 30.0% | 40.0% | 80.0% | 0.176 | 0.136 |

`cross_document`, all required documents present in top k: **k=5 30%, k=10 60%,
k=20 90%**.

The same number split by where the question came from, because the two groups
are not comparable:

| Subset | n | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|---:|
| original questions, FY24-only corpus (`20260921T165208Z`) | 90 | 51.1% | 61.1% | 72.2% |
| original questions, FY23 + FY24 corpus | 90 | 43.3% | 52.2% | 65.6% |
| new FY23 questions | 19 | 68.4% | 84.2% | 94.7% |

**The drop from 51.1% to 43.3% is year confusion.** Consecutive annual reports
are near-duplicates, and a dense embedding of "revenue in FY 2024" sits as
close to last year's page as to this year's. In all 9 questions that lost
their Recall@5 hit, FY23 pages took 3–5 of the top 5 slots; in only one did
that page state the answer. It runs both ways — on the FY23 questions, 39 of
95 top-5 slots went to FY24 pages.

**The new FY23 questions score high for a reason that is not "FY23 is easier".**
They ask headline figures that the FY24 report also prints as prior-year
comparatives, so their gold lists span both reports — any one of many pages is
a hit. Read them as a coverage check on the FY23 documents, not as evidence
that retrieval works.

**Supplying the right company and year as a filter** (an oracle taken from the
gold labels, so an upper bound for Phase 4's time parsing): company alone
48.6% Recall@5, company + year **58.7%** (R@10 68.8%, R@20 76.1%). The year
filter is worth 11 points; the 41% that still miss are a retrieval-quality
problem that no amount of query parsing fixes.

Readings from the FY24-only baseline that still hold:

**Widening k from 5 to 20 buys only 21–23 points overall.** Most misses are not
mis-ranked, they are not retrieved at all. A cross-encoder reranker can only
reorder what dense search already returned, so on this evidence reranking is
the *second* Phase 3 lever, not the first — hybrid BM25 and structure-aware
chunking come first.

**`cross_document` is the exception: 30–40% at k=5, 80% at k=20.** For comparison
questions the right chunks *are* being found and then crowded out, because one
company's document can occupy all five slots. That is a top-k or per-document
quota problem, not a retrieval-quality problem, and it is cheap to fix.

On the FY24-only corpus narrative questions retrieved best (70%), which is what
dense embeddings are good at, and the gap to lookup (50%) was the argument for
hybrid search in one number. With FY23 indexed the same narrative questions
fell to 50%, the steepest drop of any category.

## Two things the metric above does not yet capture

**Cross-document recall is scored leniently.** A question counts as a hit the
moment *any* gold page is retrieved, but `cross-05` needs all three companies
present to be answerable at all. The runner reports both — and the gap is
instructive: at k=5, recall is 40% while full document coverage is 50%, meaning
the right *documents* often reach the prompt while the right *pages* do not.
Neither number alone tells you that.

**Gold lists are exhaustive only where the fact has a distinctive figure.**
Labels were expanded by searching the corpus for each answer's figure and
keeping pages where a matching label sits within 400 characters of it — which
grew the set from 189 to 281 gold pages (441 now, with the FY23 labels) and turned `lookup-01` from a false
miss into a hit. The answer eval's first smoke run then found TCS p.76 missing
from `cross-01` and `arith-04`; adding it raised Recall@5 from 48.9% to 51.1%.
Both runs are kept in `data/eval_runs/`. Facts identified only by a percentage (`20.7%`, `3.53 per
cent`) could not be expanded that way, so a few gold lists may still be
incomplete and those questions' recall is understated rather than overstated.

## Answer evaluation

```bash
python backend/eval/run_answer_eval.py --label "phase2-baseline"      # ~1 day's tokens
python backend/eval/run_answer_eval.py --ids lookup-19 arith-05       # a few questions
python backend/eval/run_answer_eval.py --resume data/eval_runs/answers-<stamp>.jsonl
```

Runs every question through the same `search()` and `answer_question()` that
`POST /chat` uses, then scores the response against the question's key facts.

| Metric | Scored how | Model call? |
|---|---|---|
| correct | all `must_include` present, no `must_not_include`, not refused | no |
| complete | correct, plus every `qualifier` present | no |
| answered, wrong | gave an answer that is not correct — a plausible wrong figure, the worst outcome here | no |
| correct refusal | refusal questions: did it decline? | no |
| over-refusal | answerable questions: did it decline anyway? | no |
| cited a gold page | any citation lands on a gold page | no |
| faithfulness | share of claims the retrieved sources support — **narrative only** | yes, a judge |

**Diagnosis — which half of RAG failed.** Every answerable question is put in
one of four buckets:

| | answer correct | answer wrong or refused |
|---|---|---|
| **gold page retrieved** | `correct` | `generation_failure` — fix the prompt |
| **not retrieved** | `correct_unlabelled_page` — expand gold | `retrieval_failure` — fix search |

For comparisons, "retrieved" means every company's gold page came back — one
company's page alone cannot answer "which was higher?".

**Cost and pacing.** ~1,750 tokens per question, measured; ~3,500 for a
narrative question including its judge call. A full run is roughly one day's
200K budget. The runner paces itself under 7,000 tokens/minute (Groq's limit is
8,000/min), writes each result as it finishes, and stops cleanly on an error
such as the daily limit — **an error is never scored as a wrong answer**.
`--resume` continues, refusing if the model or retrieval settings changed.

**What the smoke run showed**, before any full run:

- `arith-05` retrieved the right page and **refused anyway** — prompt rules 4
  (report the figures, decline the calculation) and 5 (refuse if not in
  sources) compete, and the model picked 5. A generation failure.
- `lookup-35` answered "3.53 per cent" without "Core" — correct, not complete,
  exactly as in the Phase 1 spot check.
- `cross-01` refused because the Infosys page never came back — a retrieval
  failure, consistent with the 40% `cross_document` Recall@5 above.

## Corpus gaps this set cannot cover

The corpus is the FY23 and FY24 annual reports for three companies; FY22 is not
yet ingested. Two consequences:

- **No transcript questions.** project.md's `transcript_search` category needs
  earnings call transcripts, which are not ingested. The `narrative` category
  stands in — management commentary from MD&A and the CEO letter has the same
  retrieval shape (qualitative, no figures) — but it is not the same test.
  `refusal-10` and `refusal-11` currently expect refusal *because* the
  documents are absent; they become answerable questions once transcripts are
  ingested, so they double as a corpus coverage tracker.
- **`multi_year` questions are answered from the history tables inside the FY24
  reports.** Their gold lists do not yet include FY23-report pages, because
  every one of them also asks for an FY24 figure the FY23 report cannot contain. Both TCS (5-year charts on
  p18, 10-year table on p90) and Infosys (5-year table on p16) publish enough
  history for real multi-year questions. This tests retrieval over dense tables,
  which is worth measuring on its own. Questions that need the FY23 *and* FY24
  reports together — the real cross-year test — are still to be written.

## A parsing finding worth carrying into Phase 3

The rupee sign does not survive extraction consistently. In the HDFC Bank
report it comes out as `H` (`H85.8`, `H60,812.3 crore`) on some pages and `L`
or a backtick on others; the Infosys report uses a backtick (`` `577 cr ``).

This matters for Phase 3 specifically: a BM25 index will not match a query
token containing a currency symbol against these pages, so hybrid search will
look weaker than it is unless the currency glyph is normalised at parse time.
Six questions in this set (`lookup-27`, `lookup-41`, `lookup-48`, `lookup-50`,
and the CSR comparisons) sit directly on that fault line.

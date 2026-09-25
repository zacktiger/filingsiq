"""
Retrieval half of the Phase 2 eval: does search find the right page at all?

This runs NO model and spends NO tokens. That is the whole design point. The
answer half of the eval costs ~1,750 tokens per question, so a full run is
roughly all of Groq's 200K/day budget and can happen about once a day. Retrieval
metrics need only the question and its gold pages, so they can run on every
chunking, embedding or top_k change - which is the loop Phase 3 lives in.

    python backend/eval/run_retrieval_eval.py
    python backend/eval/run_retrieval_eval.py --k 5 10 20 50
    python backend/eval/run_retrieval_eval.py --label "phase3-bm25"

Each run writes data/eval_runs/<timestamp>.json with the per-question results
and the index configuration that produced them, so two runs can be diffed
rather than compared from memory.

--- On the three metrics -----------------------------------------------------

Recall@k  - was any gold page in the top k? This is the metric that decides
            whether an answer is POSSIBLE. If the page never reaches the
            prompt, no amount of prompt engineering recovers it.
MRR@k     - 1/rank of the first gold page. Rewards putting it first rather than
            fifth, which matters because top_k=5 is the live setting.
nDCG@k    - credits finding SEVERAL gold pages high up. Most useful for the
            multi_year and cross_document questions, where one chunk is not
            enough to answer.

Both MRR and nDCG are computed over the LARGEST requested cutoff, so both move
if --k changes - a gold page at rank 15 contributes 1/15 to MRR@20 and 0 to
MRR@10. They are reported as MRR@k / nDCG@k for that reason, and two runs are
only comparable when they used the same largest k.

Recall is the headline. MRR and nDCG explain *why* a change helped: a jump in
recall means retrieval found new things; a jump in MRR alone means it merely
reordered what it already had.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DATA_DIR, get_settings  # noqa: E402
from app.rag.store import build_filter, get_store, with_inferred_filters  # noqa: E402
from app.schemas import ChatRequest  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RUNS_DIR = DATA_DIR / "eval_runs"

DEFAULT_KS = (5, 10, 20)


def load_questions() -> list[dict]:
    """Load the question set, dropping refusal questions.

    Refusal questions have no gold pages by definition - the answer is not in
    the corpus - so there is nothing for a retrieval metric to score. They are
    scored by the answer-half eval, which checks that the system declined.
    """
    questions = []
    with QUESTIONS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            question = json.loads(line)
            if question.get("expected_refusal") is None:
                questions.append(question)
    return questions


def gold_pages(question: dict) -> set[tuple[str, int]]:
    """Every (document, page) that states the answer.

    A question is a hit if ANY of them is retrieved, because they are
    alternative correct sources, not a set that must all be found. Labelling
    all of them matters: an incomplete gold list reports a correct retrieval
    as a miss, which is how a baseline ends up pessimistic for no reason.
    """
    return {(entry["doc"], page) for entry in question["gold"] for page in entry["pages"]}


def retrieved_pages(
    store, question: str, k: int, year_filter: bool = True
) -> list[tuple[str, int]]:
    """Run one search and return its (document, page) list in rank order.

    Filters go through the same with_inferred_filters() as store.search(), so
    the eval measures what the API actually does. year_filter=False reproduces
    the unfiltered baseline.

    Paths are normalised to forward slashes because `source_path` is built from
    a Windows path at ingest time but the question set is written with POSIX
    separators. Comparing them raw silently scores every question as a miss.
    """
    request = ChatRequest(question=question)
    if year_filter:
        request = with_inferred_filters(request)
    hits = store.similarity_search(question, k=k, filter=build_filter(request))
    return [
        (hit.metadata["source_path"].replace("\\", "/"), hit.metadata["page"])
        for hit in hits
    ]


def reciprocal_rank(retrieved: list, gold: set) -> float:
    """1/rank of the first gold page, or 0 if none was retrieved."""
    for position, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: list, gold: set, k: int) -> float:
    """nDCG with binary relevance over the top k.

    The ideal ranking puts as many gold pages at the top as exist, capped at k -
    so a question with two gold pages is not penalised for failing to fill ten
    slots it could never fill.
    """
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(retrieved[:k], start=1)
        if item in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def documents_covered(retrieved: list, question: dict, k: int) -> bool:
    """Whether every document the question needs is represented in the top k.

    Recall@k is lenient for comparisons: "which of the three spent most on CSR"
    counts as a hit the moment ONE company's page appears, even though the
    question cannot be answered without all three. This stricter measure is
    reported alongside it for cross_document questions so the gap is visible
    rather than flattering.
    """
    needed = {entry["doc"] for entry in question["gold"]}
    seen = {doc for doc, _ in retrieved[:k]}
    return needed <= seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="cutoffs to report (default: 5 10 20)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="short name for this run, recorded in the results file",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="print results without writing a file"
    )
    parser.add_argument(
        "--no-year-filter",
        action="store_true",
        help="do not infer a fiscal-year filter from the question (the pre-Phase-4 baseline)",
    )
    args = parser.parse_args()

    ks = sorted(set(args.k))
    max_k = max(ks)

    questions = load_questions()
    settings = get_settings()
    store = get_store()

    per_question = []
    # [hits per k..., reciprocal rank sum, ndcg sum, n]
    totals: dict[str, dict] = defaultdict(
        lambda: {"hits": dict.fromkeys(ks, 0), "rr": 0.0, "ndcg": 0.0, "n": 0}
    )
    coverage = {"hits": dict.fromkeys(ks, 0), "n": 0}

    for question in questions:
        gold = gold_pages(question)
        retrieved = retrieved_pages(
            store, question["question"], max_k, year_filter=not args.no_year_filter
        )

        hits = {k: any(item in gold for item in retrieved[:k]) for k in ks}
        rr = reciprocal_rank(retrieved, gold)
        ndcg = ndcg_at_k(retrieved, gold, max_k)

        for bucket in (totals[question["category"]], totals["ALL"]):
            for k in ks:
                bucket["hits"][k] += hits[k]
            bucket["rr"] += rr
            bucket["ndcg"] += ndcg
            bucket["n"] += 1

        if question["category"] == "cross_document":
            coverage["n"] += 1
            for k in ks:
                coverage["hits"][k] += documents_covered(retrieved, question, k)

        per_question.append(
            {
                "id": question["id"],
                "category": question["category"],
                "difficulty": question["difficulty"],
                "hit": {str(k): hits[k] for k in ks},
                "reciprocal_rank": round(rr, 4),
                "ndcg": round(ndcg, 4),
                "retrieved": [f"{doc}#{page}" for doc, page in retrieved[:max_k]],
            }
        )

    header = f"{'category':22} {'n':>3}  " + "  ".join(f"R@{k:<4}" for k in ks)
    header += f"  MRR@{max_k}  nDCG@{max_k}"
    year_mode = "off" if args.no_year_filter else "inferred"
    print(
        f"\n{settings.embedding_model}  |  year filter: {year_mode}  |  "
        f"top_k as evaluated: {ks}"
    )
    print(header)
    print("-" * len(header))
    for category in sorted(totals, key=lambda c: (c != "ALL", c)):
        bucket = totals[category]
        n = bucket["n"]
        row = f"{category:22} {n:>3}  "
        row += "  ".join(f"{bucket['hits'][k] / n:5.1%}" for k in ks)
        row += f"  {bucket['rr'] / n:.3f}  {bucket['ndcg'] / n:.3f}"
        print(row)

    if coverage["n"]:
        print(
            f"\ncross_document, all required documents present in top k: "
            + ", ".join(
                f"k={k} {coverage['hits'][k] / coverage['n']:.0%}" for k in ks
            )
        )

    smallest_k = ks[0]
    misses = [r["id"] for r in per_question if not r["hit"][str(smallest_k)]]
    print(f"\nRecall@{smallest_k} misses ({len(misses)}):")
    print("  " + ", ".join(misses) if misses else "  none")

    if args.no_save:
        return 0

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RUNS_DIR / f"{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "label": args.label,
                # Recorded so a later run can be compared honestly: a recall
                # change means nothing if the index underneath it also changed.
                "embedding_model": settings.embedding_model,
                "collection": settings.collection_name,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "year_filter": not args.no_year_filter,
                "ks": ks,
                "questions": len(questions),
                "summary": {
                    category: {
                        "n": bucket["n"],
                        **{
                            f"recall@{k}": round(bucket["hits"][k] / bucket["n"], 4)
                            for k in ks
                        },
                        f"mrr@{max_k}": round(bucket["rr"] / bucket["n"], 4),
                        f"ndcg@{max_k}": round(bucket["ndcg"] / bucket["n"], 4),
                    }
                    for category, bucket in totals.items()
                },
                "per_question": per_question,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path.relative_to(DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

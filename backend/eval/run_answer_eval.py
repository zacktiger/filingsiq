"""
Answer half of the Phase 2 eval: given what retrieval found, did the system
say the right thing?

This one DOES spend tokens - ~1,750 per question on the real corpus, so a full
run over 102 questions uses roughly all of Groq's 200K/day free tier. That is
why it is a separate script from run_retrieval_eval.py, which is free and runs
constantly. Run this one deliberately: once for a baseline, then when comparing
answers. If the daily limit stops it part way, --resume continues next day.

    python backend/eval/run_answer_eval.py --label "phase2-baseline"
    python backend/eval/run_answer_eval.py --ids lookup-19 arith-05 refusal-07
    python backend/eval/run_answer_eval.py --resume data/eval_runs/answers-<stamp>.jsonl

It calls the same search() and answer_question() that POST /chat calls, so it
measures the real system, not a copy of it.

--- What is scored, and how ------------------------------------------------

correct            every must_include fact is in the answer, nothing from
                   must_not_include is, and the system did not refuse
complete           correct, AND every qualifier is present ("Core" NIM)
answered, wrong    the system gave an answer and it was not correct. The
                   worst outcome this project has: a plausible wrong figure
                   is more harmful than a refusal
correct refusal    for the 12 refusal questions, did it decline?
over-refusal       for answerable questions, did it decline anyway?
citation accuracy  did the answer cite a gold page? Understated where gold
                   lists are incomplete - see backend/eval/README.md
faithfulness       narrative questions only: share of the answer's claims
                   that the retrieved sources actually state (LLM judge)

--- The diagnosis column: which half of RAG failed --------------------------

Every answerable question is sorted into one of four buckets by crossing
"was a gold page retrieved?" with "was the answer correct?":

    correct                   retrieved it, answered it
    generation failure        retrieved it, answered WRONG -> fix the prompt
    retrieval failure         never retrieved it           -> fix search
    correct, unlabelled page  answered correctly from a page not in gold
                              -> usually a gold list that needs expanding

This is the point of evaluating retrieval and generation separately. An
end-to-end score says only "wrong"; the diagnosis says which stage to fix.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Filings text contains the rupee sign, and the Windows console defaults to a
# code page that cannot encode it. Same fix, and same reasoning, as ask.py.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402

from app.config import DATA_DIR, get_settings  # noqa: E402
from app.rag.answer import answer_question, format_sources, get_model  # noqa: E402
from app.rag.embed import get_embeddings  # noqa: E402
from app.rag.store import search  # noqa: E402
from app.schemas import ChatRequest  # noqa: E402
from scoring import missing_terms, present_terms  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RUNS_DIR = DATA_DIR / "eval_runs"

# Groq's free tier allows 8K tokens per MINUTE as well as 200K per day, and the
# per-minute limit is the one a batch run hits first. answer.py's rate limiter
# caps REQUESTS (0.4/s), not tokens - and a question here costs ~1,750 tokens
# and answers in ~2s, so an unpaced run would try to spend ~50K tokens/minute.
#
# So after each question the runner waits until that question's tokens fit
# inside the per-minute budget: 1,750 tokens at 7,000/min is 15s, a narrative
# question with a judge call (~3,500) is 30s. Pacing on tokens actually spent,
# rather than a fixed delay, stays correct when questions differ in size.
DEFAULT_TOKENS_PER_MINUTE = 7_000

# Measured on the real corpus (smoke run, 2026-09-21): input 1,325-1,805
# tokens, output 87-324. Five 1,000-character chunks from a real filing are far
# larger than the synthetic fixture's. Used only for the up-front estimate.
MEASURED_TOKENS_PER_QUESTION = 1_750

JUDGE_PROMPT = """\
You check whether an answer is supported by source excerpts.

Split the ANSWER into its individual factual claims. For each claim, decide \
whether the SOURCES state it. A claim is supported only if the sources say it; \
general knowledge does not count, and a claim that goes beyond what the sources \
say is not supported.

Reply with JSON only - no prose, no code fence - in exactly this shape:
{"claims": [{"claim": "<the claim, briefly>", "supported": true}]}"""


def load_questions(ids: list[str] | None, category: str | None) -> list[dict]:
    """Load the question set, optionally narrowed to ids or one category."""
    questions = []
    with QUESTIONS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                questions.append(json.loads(line))
    if ids:
        wanted = set(ids)
        unknown = wanted - {q["id"] for q in questions}
        if unknown:
            raise SystemExit(f"Unknown question id(s): {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q["id"] in wanted]
    if category:
        questions = [q for q in questions if q["category"] == category]
    return questions


def run_config() -> dict:
    """Everything that could change an answer, recorded with the results.

    Two answer runs are only comparable if these match. A resumed run checks
    them, because stitching together answers from two different models or
    retrieval settings would produce a baseline that describes neither.
    """
    settings = get_settings()
    return {
        "answer_model": settings.answer_model,
        "answer_effort": settings.answer_effort,
        "answer_temperature": settings.answer_temperature,
        "answer_max_tokens": settings.answer_max_tokens,
        "top_k": settings.top_k,
        "embedding_model": settings.embedding_model,
        "collection": settings.collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def page_ref(source_path: str, page: int) -> str:
    """One (document, page) as a comparable string.

    source_path is written from a Windows path at ingest time; the question set
    uses forward slashes. Normalising here is what stops every comparison from
    silently failing - the same trap as in run_retrieval_eval.py.
    """
    return f"{source_path.replace(chr(92), '/')}#{page}"


def judge_faithfulness(chunks, answer: str) -> tuple[float | None, list, str | None]:
    """Ask the model which of the answer's claims the sources support.

    Used on narrative questions only. Their answers are free text, so a string
    check can confirm a key figure is present but cannot tell whether the REST
    of the paragraph was invented. That is the one job a judge does here.

    Known weakness, stated rather than hidden: the judge is the same model that
    wrote the answer, and models tend to rate their own output generously. Read
    faithfulness as an upper bound. A different judge model would fix it, and
    would need a second provider or a second slice of the same token budget.
    """
    reply = get_model().invoke(
        [
            ("system", JUDGE_PROMPT),
            ("human", f"SOURCES\n\n{format_sources(chunks)}\n\nANSWER\n{answer}"),
        ]
    )
    raw = reply.content if isinstance(reply.content, str) else str(reply.content)
    start, end = raw.find("{"), raw.rfind("}")
    try:
        claims = json.loads(raw[start : end + 1])["claims"]
        supported = sum(1 for claim in claims if claim.get("supported") is True)
        score = supported / len(claims) if claims else None
        return score, claims, None
    except (ValueError, KeyError, TypeError) as exc:
        # Recorded, not raised: one malformed judge reply should not discard a
        # run that has already spent tokens on every question before it.
        return None, [], f"judge reply was not valid JSON: {exc}"


def score(question: dict, response, chunks) -> dict:
    """Compare one response against its question's key facts and gold pages."""
    gold = {page_ref(e["doc"], p) for e in question["gold"] for p in e["pages"]}
    retrieved = [page_ref(c.metadata["source_path"], c.metadata["page"]) for c in chunks]
    cited = [page_ref(c.source_path, c.page) for c in response.citations]

    record = {
        "refused": response.refused,
        "retrieved": retrieved,
        "cited": cited,
    }

    if question["expected_refusal"] is not None:
        record["correct"] = response.refused
        return record

    must = question["must_include"]
    answer = "" if response.refused else response.answer
    missing = missing_terms(must, answer) if answer else list(must)
    violations = present_terms(question["must_not_include"], answer) if answer else []
    correct = bool(answer) and not missing and not violations
    qualifiers_missing = missing_terms(question["qualifiers"], answer) if answer else []

    # A hit means EVERY document the question needs had a gold page retrieved.
    # For a single-company question that is just "any gold page came back".
    # For a comparison it is stricter, and has to be: if TCS's margin page is
    # retrieved but Infosys's is not, the question cannot be answered, and
    # blaming the refusal on generation would send you to fix the wrong stage.
    retrieval_hit = all(
        any(page_ref(entry["doc"], page) in retrieved for page in entry["pages"])
        for entry in question["gold"]
    )
    if correct:
        diagnosis = "correct" if retrieval_hit else "correct_unlabelled_page"
    else:
        diagnosis = "generation_failure" if retrieval_hit else "retrieval_failure"

    cited_in_gold = [ref for ref in cited if ref in gold]
    record.update(
        {
            "correct": correct,
            "complete": correct and not qualifiers_missing,
            "missing": missing,
            "qualifiers_missing": qualifiers_missing,
            "violations": violations,
            "retrieval_hit": retrieval_hit,
            "diagnosis": diagnosis,
            "cited_gold": bool(cited_in_gold),
            "citation_precision": (len(cited_in_gold) / len(cited)) if cited else None,
        }
    )
    return record


def evaluate_one(question: dict, use_judge: bool) -> dict:
    """Run one question through the live pipeline and score it."""
    started = time.monotonic()
    with get_usage_metadata_callback() as usage:
        request = ChatRequest(question=question["question"])
        chunks = search(request)
        response = answer_question(request, chunks)

        record = {
            "id": question["id"],
            "category": question["category"],
            "difficulty": question["difficulty"],
            "question": question["question"],
            "answer": response.answer,
            **score(question, response, chunks),
        }

        if use_judge and question["category"] == "narrative" and not response.refused:
            faithfulness, claims, error = judge_faithfulness(chunks, response.answer)
            record.update(
                {"faithfulness": faithfulness, "judge_claims": claims, "judge_error": error}
            )

    tokens = {"input": 0, "output": 0, "total": 0}
    for model_usage in usage.usage_metadata.values():
        tokens["input"] += model_usage.get("input_tokens", 0)
        tokens["output"] += model_usage.get("output_tokens", 0)
        tokens["total"] += model_usage.get("total_tokens", 0)
    record["tokens"] = tokens
    record["latency_s"] = round(time.monotonic() - started, 2)
    return record


def open_run(resume: str | None, label: str) -> tuple[Path, set[str]]:
    """Start a new results file, or reopen a partial one.

    Results are appended one line per question AS THEY FINISH, so a run killed
    by the daily token limit at question 60 keeps its 60 answers. Those answers
    cost tokens; throwing them away and starting over would spend the budget
    twice for one baseline.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config = run_config()

    if resume:
        path = Path(resume)
        config_path = path.with_suffix(".config.json")
        if not path.exists() or not config_path.exists():
            raise SystemExit(f"Cannot resume: {path} or its .config.json is missing.")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        changed = {k for k in config if saved.get("config", {}).get(k) != config[k]}
        if changed:
            raise SystemExit(
                "Cannot resume: settings changed since this run started "
                f"({', '.join(sorted(changed))}). Mixing answers from two "
                "configurations would produce a baseline that describes neither. "
                "Start a new run instead."
            )
        done = {
            json.loads(line)["id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        return path, done

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"answers-{stamp}.jsonl"
    path.with_suffix(".config.json").write_text(
        json.dumps(
            {"started_at": datetime.now(timezone.utc).isoformat(), "label": label, "config": config},
            indent=2,
        ),
        encoding="utf-8",
    )
    path.touch()
    return path, set()


def summarise(records: list[dict]) -> dict:
    """Aggregate per-question records into the numbers worth reporting."""
    answerable = [r for r in records if r["category"] != "refusal"]
    refusals = [r for r in records if r["category"] == "refusal"]
    answered = [r for r in answerable if not r["refused"]]

    def rate(items, key):
        return round(sum(1 for r in items if r.get(key)) / len(items), 4) if items else None

    by_category = {}
    for category in sorted({r["category"] for r in answerable}):
        rows = [r for r in answerable if r["category"] == category]
        by_category[category] = {
            "n": len(rows),
            "correct": rate(rows, "correct"),
            "complete": rate(rows, "complete"),
            "answered_wrong": round(
                sum(1 for r in rows if not r["refused"] and not r["correct"]) / len(rows), 4
            ),
            "refused": rate(rows, "refused"),
        }

    precisions = [r["citation_precision"] for r in answered if r["citation_precision"] is not None]
    faithfulness = [r["faithfulness"] for r in records if r.get("faithfulness") is not None]
    tokens = [r["tokens"]["total"] for r in records]

    return {
        "questions": len(records),
        "answerable": {
            "n": len(answerable),
            "correct": rate(answerable, "correct"),
            "complete": rate(answerable, "complete"),
            "answered_wrong": round(
                sum(1 for r in answerable if not r["refused"] and not r["correct"])
                / len(answerable),
                4,
            )
            if answerable
            else None,
            "over_refusal": rate(answerable, "refused"),
            "arithmetic_violations": sum(1 for r in answerable if r.get("violations")),
        },
        "by_category": by_category,
        "refusal": {"n": len(refusals), "correct_refusal": rate(refusals, "correct")},
        "citations": {
            "answered": len(answered),
            "cited_a_gold_page": rate(answered, "cited_gold"),
            "mean_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
            "answers_with_no_citation": sum(1 for r in answered if not r["cited"]),
        },
        "diagnosis": dict(Counter(r["diagnosis"] for r in answerable)),
        "faithfulness": {
            "judged": len(faithfulness),
            "mean": round(sum(faithfulness) / len(faithfulness), 4) if faithfulness else None,
        },
        "cost": {
            "total_tokens": sum(tokens),
            "mean_tokens_per_question": round(sum(tokens) / len(tokens)) if tokens else None,
            "mean_latency_s": round(sum(r["latency_s"] for r in records) / len(records), 2)
            if records
            else None,
        },
    }


def print_summary(summary: dict) -> None:
    def pct(value):
        return "  -  " if value is None else f"{value:5.1%}"

    print(f"\n{'category':22} {'n':>3}  correct  complete  answered-wrong  refused")
    print("-" * 72)
    for category, row in summary["by_category"].items():
        print(
            f"{category:22} {row['n']:>3}  {pct(row['correct'])}    {pct(row['complete'])}     "
            f"{pct(row['answered_wrong'])}        {pct(row['refused'])}"
        )
    a = summary["answerable"]
    print("-" * 72)
    print(
        f"{'answerable, all':22} {a['n']:>3}  {pct(a['correct'])}    {pct(a['complete'])}     "
        f"{pct(a['answered_wrong'])}        {pct(a['over_refusal'])}"
    )

    r = summary["refusal"]
    c = summary["citations"]
    f = summary["faithfulness"]
    cost = summary["cost"]
    print(f"\ncorrect refusal rate      {pct(r['correct_refusal'])}  (of {r['n']} refusal questions)")
    print(f"arithmetic violations     {a['arithmetic_violations']}")
    print(f"cited a gold page         {pct(c['cited_a_gold_page'])}  (of {c['answered']} answered)")
    print(f"answers with no citation  {c['answers_with_no_citation']}")
    if f["judged"]:
        print(f"faithfulness (narrative)  {pct(f['mean'])}  (judged {f['judged']}, same-model judge: upper bound)")
    print("\ndiagnosis of answerable questions:")
    for bucket in ("correct", "generation_failure", "retrieval_failure", "correct_unlabelled_page"):
        print(f"  {bucket:26} {summary['diagnosis'].get(bucket, 0)}")
    print(
        f"\ntokens: {cost['total_tokens']:,} total, ~{cost['mean_tokens_per_question']} per question; "
        f"mean latency {cost['mean_latency_s']}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer-quality eval (spends tokens).")
    parser.add_argument("--ids", nargs="+", help="run only these question ids")
    parser.add_argument("--category", help="run only one category")
    parser.add_argument("--resume", help="append to a partial answers-*.jsonl run")
    parser.add_argument("--label", default="", help="short name recorded with the run")
    parser.add_argument(
        "--tokens-per-minute",
        type=int,
        default=DEFAULT_TOKENS_PER_MINUTE,
        help=f"pace the run under this budget; Groq's free tier is 8,000 (default {DEFAULT_TOKENS_PER_MINUTE:,})",
    )
    parser.add_argument("--no-judge", action="store_true", help="skip the faithfulness judge")
    args = parser.parse_args()

    if not get_settings().groq_api_key:
        raise SystemExit("GROQ_API_KEY is not set in .env - the answer eval needs it.")

    questions = load_questions(args.ids, args.category)
    path, done = open_run(args.resume, args.label)
    todo = [q for q in questions if q["id"] not in done]

    estimate = len(todo) * MEASURED_TOKENS_PER_QUESTION
    print(
        f"{len(todo)} question(s) to run ({len(done)} already done) -> {path.name}\n"
        f"Estimated cost ~{estimate:,} tokens of the 200,000/day free tier; "
        f"~{estimate / args.tokens_per_minute:.0f} min paced at "
        f"{args.tokens_per_minute:,} tokens/minute.\n"
    )

    # Load the embedding model before the clock starts. Otherwise the first
    # question's latency includes ~50s of loading weights and skews the mean.
    # Deliberately NOT warming the LLM - that would spend tokens on nothing.
    get_embeddings().embed_query("warmup")

    stopped_early = False
    for index, question in enumerate(todo, start=1):
        started = time.monotonic()
        try:
            record = evaluate_one(question, use_judge=not args.no_judge)
        except KeyboardInterrupt:
            stopped_early = True
            break
        except Exception as exc:  # noqa: BLE001 - stop cleanly, keep what was paid for
            # Most likely the daily token limit. Recording this as a wrong
            # answer would corrupt the baseline, so the run stops instead and
            # everything answered so far is kept for --resume.
            print(f"\nStopped at {question['id']}: {type(exc).__name__}: {exc}")
            stopped_early = True
            break

        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        verdict = record.get("diagnosis") or ("correct" if record["correct"] else "WRONG")
        if question["category"] == "refusal":
            verdict = "refused (correct)" if record["refused"] else "ANSWERED (should refuse)"
        print(
            f"[{index:>3}/{len(todo)}] {question['id']:13} {verdict:26} "
            f"{record['tokens']['total']:>5} tok  {record['latency_s']:>5.1f}s"
        )

        # Wait until this question's tokens fit in the per-minute budget.
        needed = record["tokens"]["total"] / args.tokens_per_minute * 60
        elapsed = time.monotonic() - started
        if index < len(todo) and elapsed < needed:
            time.sleep(needed - elapsed)

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        print("No results recorded.")
        return 1

    summary = summarise(records)
    print_summary(summary)

    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {path.relative_to(DATA_DIR.parent)} and {summary_path.name}")

    if stopped_early:
        print(
            "\nRun incomplete. Continue it (after the daily limit resets, if that "
            f"was the cause) with:\n  python backend/eval/run_answer_eval.py "
            f"--resume {path.relative_to(DATA_DIR.parent).as_posix()}"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

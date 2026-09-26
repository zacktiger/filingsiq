"""
Stage 6 of the pipeline: retrieved chunks + question -> a cited answer.

Two project goals are enforced here rather than hoped for:

  * Every claim cites a document and page. The model marks its sources as [S1],
    [S2]; those markers are then resolved back against the chunks that were
    actually retrieved. A citation the model invents cannot survive that step,
    because there is nothing to resolve it to.

  * The system refuses instead of guessing. Two sentinel replies -
    INSUFFICIENT_CONTEXT and REFUSE_OUT_OF_SCOPE - let refusal be detected
    exactly, rather than by pattern-matching apologetic prose. Phase 2's eval
    scores "correct refusal rate" off this.
"""

import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, Citation

# Sentinels the model emits instead of an answer. Uppercase and underscored so
# they cannot plausibly appear in an ordinary sentence about a filing.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"
OUT_OF_SCOPE = "REFUSE_OUT_OF_SCOPE"

SYSTEM_PROMPT = f"""\
You are FilingsIQ, an assistant that answers questions about Indian listed \
companies using ONLY the filings excerpts provided to you.

RULES

1. Use only the numbered SOURCES below. Never use outside knowledge about these \
companies, even if you are confident it is correct.
2. Cite every factual claim with the source marker it came from, using plain \
ASCII square brackets exactly like [S1] or [S2][S3]. Do not use any other \
bracket characters. A sentence stating a fact without a marker is a mistake.
3. Quote figures exactly as they appear, including units and currency \
(for example "Rs. 12,345 crore"). Do not convert, round, or rescale them.
4. Do NOT perform arithmetic across sources - no growth rates, ratios, CAGRs, \
or margins that are not printed in the excerpts. If a question needs a \
calculation, report the underlying figures you can cite and say the calculation \
is not available.
5. If the SOURCES do not contain the answer, reply with exactly \
{INSUFFICIENT} and nothing else. Do not apologise or speculate.
6. If the question asks you to predict future performance, or for investment, \
buying, or selling advice, reply with exactly {OUT_OF_SCOPE} and nothing else.

Be concise. Lead with the answer, then the supporting detail."""


@lru_cache
def get_model() -> BaseChatModel:
    """Build the answer model once per process.

    Runs on Groq's free tier (gpt-oss-120b). Three settings here are
    load-bearing rather than incidental:

    * temperature=0. ChatGroq defaults to 0.7. This task copies figures out of
      a filing verbatim, so sampling randomness is precisely how a correct
      "Rs. 48,250 crore" turns into a plausible, unverifiable "Rs. 48,520
      crore". Nothing about this workload wants variety.

    * A client-side rate limiter. The free tier allows 30 requests/minute, and
      the Phase 2 eval fires 80-100 questions in a batch. Throttling locally
      makes that queue instead of failing half way through with HTTP 429s.

    * max_tokens. The free tier's real ceiling is 200,000 tokens per DAY, which
      the eval runs also draw on, so answers are capped rather than unbounded.
    """
    settings = get_settings()

    limiter = InMemoryRateLimiter(
        requests_per_second=settings.requests_per_second,
        # Check often enough that a queued request starts promptly once a slot
        # frees, rather than waiting out a long polling interval.
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    if settings.answer_provider == "gemini":
        # Imported here so the deployed Space, which only uses Groq, does not
        # need the Gemini package installed.
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.google_api_key,
            # Same rule as Groq: figures are copied verbatim, so no sampling.
            # Gemini accepts temperature=0 (checked with a live call); the
            # opposite of the claude-opus-5 case in CLAUDE.md.
            temperature=settings.answer_temperature,
            # Gemini counts its thinking tokens against the output cap, so the
            # Groq-sized 1024 could cut an answer off mid-sentence.
            max_tokens=4 * settings.answer_max_tokens,
            rate_limiter=limiter,
            max_retries=3,
        )

    kwargs = {}
    # gpt-oss reasons before answering. Blank in config disables it, so the
    # setting can be turned off without editing code if it proves unhelpful.
    if settings.answer_effort:
        kwargs["reasoning_effort"] = settings.answer_effort

    return ChatGroq(
        model=settings.answer_model,
        api_key=settings.groq_api_key,
        temperature=settings.answer_temperature,
        max_tokens=settings.answer_max_tokens,
        rate_limiter=limiter,
        # Retry transient failures, including a 429 that slips past the limiter.
        max_retries=3,
        **kwargs,
    )


def format_sources(chunks: list[Document]) -> str:
    """Render retrieved chunks as a numbered SOURCES block.

    Each source is labelled with its company, fiscal year, document type and
    page, so the model can attribute a claim precisely - and so a reader can
    check it. The [S1] marker is the handle the model cites and that
    build_citations() resolves.
    """
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        label = (
            f"{meta.get('company_name') or meta.get('company')} | "
            f"{meta.get('fiscal_year')} | {meta.get('doc_type')} | "
            f"page {meta.get('page')}"
        )
        blocks.append(f"[S{index}] {label}\n{chunk.page_content}")
    return "\n\n".join(blocks)


# Models do not reliably emit the ASCII brackets the prompt asks for. Observed
# from gpt-oss-120b in this project: fullwidth CJK brackets (U+3010 / U+3011),
# i.e. the answer said 【S1】 and a strict r"\[S(\d+)\]" silently matched nothing,
# so a citation the model HAD supplied was dropped. Parentheses, lowercase "s",
# internal spacing, and several sources in one bracket ("[S1, S2]") also occur.
#
# So the prompt asks for one form and the parser tolerates the variants. It
# still only accepts numbers that map to a retrieved chunk, so this loosens the
# syntax accepted, never the verification.
_MARKER_GROUP = re.compile(r"[\[(【]([^\[\]()【】]{0,40})[\])】]")
_SOURCE_NUM = re.compile(r"S\s*(\d+)", re.IGNORECASE)


def cited_indexes(answer: str, source_count: int) -> list[int]:
    """Find which [Sn] markers the answer actually used.

    Markers outside the valid range are discarded. This is the verification
    step: a hallucinated [S9] against 5 retrieved sources is dropped rather
    than shown to the user as though it were real provenance.
    """
    found: set[int] = set()
    for group in _MARKER_GROUP.findall(answer):
        for number in _SOURCE_NUM.findall(group):
            found.add(int(number))
    return sorted(n for n in found if 1 <= n <= source_count)


def build_citations(answer: str, chunks: list[Document]) -> list[Citation]:
    """Turn the markers used in the answer into Citation objects."""
    citations = []
    for n in cited_indexes(answer, len(chunks)):
        meta = chunks[n - 1].metadata
        citations.append(
            Citation(
                company=meta["company"],
                fiscal_year=meta["fiscal_year"],
                doc_type=meta["doc_type"],
                page=meta["page"],
                source_path=meta["source_path"],
                quote=chunks[n - 1].page_content,
            )
        )
    return citations


def answer_question(request: ChatRequest, chunks: list[Document]) -> ChatResponse:
    """Generate a grounded answer from retrieved chunks."""
    # Retrieval found nothing - no reason to spend a model call.
    if not chunks:
        return ChatResponse(
            answer=(
                "I could not find anything about that in the filings I have indexed."
            ),
            citations=[],
            refused=True,
        )

    user_prompt = f"SOURCES\n\n{format_sources(chunks)}\n\nQUESTION\n{request.question}"

    reply = get_model().invoke(
        [("system", SYSTEM_PROMPT), ("human", user_prompt)]
    )
    # .content is a string for a plain text reply, but can be a list of blocks
    # when the model returns structured content, so normalise before parsing.
    # Gemini always returns a list of blocks; str() of that list put the
    # literal "[{'type': 'text', ...}]" into the answer. Keep only the text.
    raw = reply.content if isinstance(reply.content, str) else "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in reply.content
    )
    raw = raw.strip()

    if INSUFFICIENT in raw:
        return ChatResponse(
            answer=(
                "The filings I have indexed do not contain the answer to that "
                "question."
            ),
            citations=[],
            refused=True,
        )

    if OUT_OF_SCOPE in raw:
        return ChatResponse(
            answer=(
                "I answer questions about what is stated in past filings. I do not "
                "forecast future performance or give investment advice."
            ),
            citations=[],
            refused=True,
        )

    return ChatResponse(
        answer=raw,
        citations=build_citations(raw, chunks),
        refused=False,
    )

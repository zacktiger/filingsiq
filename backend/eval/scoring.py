"""
How an answer is checked against a question's key facts.

Shared by validate_questions.py and run_answer_eval.py so the two can never
disagree about what "the answer contains 2,40,893" means.

Each non-refusal question lists the facts a correct answer must state:

    must_include       every term must appear, or the answer is WRONG
    qualifiers         every term should appear, or the answer is INCOMPLETE
                       ("3.53 per cent" vs "Core NIM 3.53 per cent")
    must_not_include   no term may appear - used to catch arithmetic the prompt
                       forbids ("87,223" is TCS minus Infosys revenue)

A term may list alternatives separated by "|" - "b s r|bsr" matches either.

Why explicit key facts instead of an LLM judge for correctness: a judge is a
second model whose verdicts you then have to trust, and it would roughly double
the token cost of a run. Every fact in this corpus that a question asks about
is a figure or a name, so a string check is both sufficient and something you
can verify by eye. The judge is kept for the one thing strings cannot check -
faithfulness of free-text narrative answers.
"""

import re

# A number as it appears in a filing: Indian grouping (2,40,893), western
# grouping (240,893), or plain, with an optional decimal part.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> set[float]:
    """Every number in the text, with digit grouping removed.

    Comparing as numbers rather than strings is what makes "2,40,893" (Indian
    grouping, as the filings print it) equal "240,893" (as a model often
    rewrites it), and "46.0" equal "46". It also stops "46" from matching
    inside "146,463", which a substring check would get wrong.
    """
    return {round(float(match.replace(",", "")), 4) for match in _NUMBER.findall(text)}


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace, so layout differences do not matter."""
    return " ".join(text.lower().replace("’", "'").split())


def _is_numeric(term: str) -> bool:
    return bool(re.fullmatch(r"\d[\d,]*(?:\.\d+)?", term.strip()))


def term_present(term: str, text: str) -> bool:
    """Whether any alternative of a term appears in the text.

    Numeric alternatives are compared as numbers; anything else is a
    case-insensitive substring. "per cent" versus "%" never matters, because
    only the number itself is compared.
    """
    numbers = None
    normalised = _normalise(text)
    for alternative in term.split("|"):
        alternative = alternative.strip()
        if not alternative:
            continue
        if _is_numeric(alternative):
            if numbers is None:
                numbers = _numbers_in(text)
            if round(float(alternative.replace(",", "")), 4) in numbers:
                return True
        elif _normalise(alternative) in normalised:
            return True
    return False


def missing_terms(terms: list[str], text: str) -> list[str]:
    """The terms from the list that do not appear in the text."""
    return [term for term in terms if not term_present(term, text)]


def present_terms(terms: list[str], text: str) -> list[str]:
    """The terms from the list that DO appear - for must_not_include checks."""
    return [term for term in terms if term_present(term, text)]

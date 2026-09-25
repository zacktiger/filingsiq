"""
Query understanding, first slice: which fiscal year is a question about?

Measured reason this exists before the rest of Phase 4: with FY23 and FY24 both
indexed, dense search cannot tell consecutive annual reports apart. They are
near-duplicates in wording, and the year is one token in an otherwise identical
passage, so "TCS revenue in FY 2024" retrieves last year's report as readily as
this year's. Supplying the right year as a filter was worth ~11 points of
Recall@5 in an oracle test (47.7% -> 58.7%).

This module only PARSES. Whether a parsed year is actually indexed is a
question about the collection, so store.py answers it - a year outside the
corpus must mean "no filter", never "filter to nothing".

Indian fiscal years run April-March and are named by the year they END in:
"FY 2023-24", "FY24", "fiscal 2024" and "2023-24" all mean the year ending
March 2024, stored as fiscal_year="FY24" (see loader.py).
"""

import re

# "FY 2024", "FY2024", "FY'24", "FY24", "FY 2023-24", "FY 2023-2024"
_FY = re.compile(r"\bFY\s*'?(\d{4}|\d{2})(?:\s*-\s*(\d{4}|\d{2}))?\b", re.IGNORECASE)
# "fiscal 2024", "fiscal year 2024", "financial year 2023-24"
_FISCAL = re.compile(
    r"\b(?:fiscal|financial)(?:\s+year)?\s+(\d{4})(?:\s*-\s*(\d{4}|\d{2}))?\b",
    re.IGNORECASE,
)
# "as of March 31, 2024", "31 March 2024", "31st March, 2024" - the last day of
# FY24, which filings use as the date every balance-sheet figure is stated at.
# Only March 31: any other date names a quarter, and quarters are not indexed.
_YEAR_END = re.compile(
    r"\b(?:March\s+31(?:st)?|31(?:st)?\s+March),?\s+(20\d{2})\b", re.IGNORECASE
)
# "2023-24" on its own - the standard way Indian filings name a fiscal year.
_RANGE = re.compile(r"\b(20\d{2})\s*-\s*(\d{2})\b")
# A bare "2024" is only a fiscal year when the question is already talking about
# fiscal years ("net profit in fiscal 2022, 2023 and 2024"). Otherwise it is a
# target date - "net zero by 2030" is not a question about FY30.
_BARE = re.compile(r"\b(20\d{2})\b")
_FISCAL_CONTEXT = re.compile(r"\b(?:FY|fiscal|financial year)", re.IGNORECASE)


def _end_year(start: str, end: str | None) -> int:
    """The calendar year a fiscal year ends in, as a 4-digit int."""
    if end is None:
        year = int(start)
        return year + 2000 if year < 100 else year
    if len(end) == 4:
        return int(end)
    # "2023-24": the end year shares the start year's century.
    century = int(start) // 100 * 100 if len(start) == 4 else 2000
    return century + int(end)


def fiscal_years_mentioned(question: str) -> list[str]:
    """Every fiscal year the question names, as sorted "FY24"-style labels."""
    years: set[int] = set()
    rest = question
    years.update(int(y) for y in _YEAR_END.findall(rest))
    rest = _YEAR_END.sub(" ", rest)
    for pattern in (_FY, _FISCAL, _RANGE):
        for match in pattern.finditer(rest):
            start, end = match.groups()
            years.add(_end_year(start, end))
        # Blank out what was consumed, so the "2023" inside "FY 2023-24" is not
        # picked up again below as a separate bare year.
        rest = pattern.sub(" ", rest)
    if _FISCAL_CONTEXT.search(question):
        years.update(int(y) for y in _BARE.findall(rest))
    return [f"FY{year % 100:02d}" for year in sorted(years)]


def infer_fiscal_year(question: str) -> str | None:
    """The one fiscal year to filter this question to, before checking the index.

    A question naming several years ("FY 2020 through FY 2024") gets the LATEST:
    a later annual report prints earlier years as comparatives and in its
    history tables, while the earlier report cannot contain the later figures.
    No year named -> None, which searches everything.
    """
    years = fiscal_years_mentioned(question)
    return years[-1] if years else None

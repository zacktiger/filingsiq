"""
Pydantic models for the API surface and for passing data between pipeline stages.

These are the project's vocabulary. `DocumentMeta` and `Citation` in particular
are what make "every answer cites the exact document and page" enforceable
rather than aspirational: page numbers are captured during parsing and carried,
unchanged, all the way to the JSON the frontend renders.
"""

from pydantic import BaseModel, Field

# --- Document metadata -------------------------------------------------------


class DocumentMeta(BaseModel):
    """Where a document came from.

    Derived from the file's path on disk - data/raw/{company}/{fiscal_year}/{doc_type}.pdf
    - and cross-checked against data/manifest.csv. Attached to every chunk so
    retrieval can filter on it and citations can name it.
    """

    company: str = Field(description="Company slug from the path, e.g. 'tcs'")
    fiscal_year: str = Field(description="Fiscal year from the path, e.g. 'FY24'")
    doc_type: str = Field(
        description="One of: annual_report, quarterly_results, earnings_call"
    )
    source_path: str = Field(description="Path relative to the data/ directory")

    # Optional because the manifest is the only place they can come from, and a
    # PDF may be indexed before its manifest row is filled in.
    company_name: str | None = Field(
        default=None, description="Full display name, e.g. 'Tata Consultancy Services'"
    )
    sector: str | None = Field(default=None, description="e.g. 'IT'")
    source_url: str | None = Field(
        default=None, description="Original BSE/NSE/IR URL the PDF was downloaded from"
    )


# --- API request / response --------------------------------------------------


class ChatRequest(BaseModel):
    """A question from the UI."""

    question: str = Field(min_length=1, max_length=2000)

    # Optional metadata filters. Phase 4 will infer these from the question
    # itself ("Infosys in FY24" -> company=infosys, fiscal_year=FY24); for now
    # the caller may pass them explicitly, and omitting them searches everything.
    company: str | None = None
    fiscal_year: str | None = None
    doc_type: str | None = None


class Citation(BaseModel):
    """One source passage backing an answer.

    `page` is 1-based, matching what a PDF reader displays, so a reader can
    verify the claim by hand. Phase 6 turns this into a link that opens the PDF
    at that page.
    """

    company: str
    fiscal_year: str
    doc_type: str
    page: int = Field(ge=1, description="1-based page number, as shown in a PDF viewer")
    source_path: str
    quote: str = Field(description="The retrieved chunk text, for verification")


class ChatResponse(BaseModel):
    """The answer returned to the UI."""

    answer: str
    citations: list[Citation]

    # True when the system declined to answer - because retrieval found nothing
    # relevant, or the question asked for a prediction or investment advice.
    # Surfaced explicitly so the frontend can style refusals differently, and so
    # the Phase 2 eval can score "correct refusal rate" without parsing prose.
    refused: bool = False

"""
Generate a synthetic filing so the pipeline can be verified with no real PDFs.

    python backend/scripts/make_test_pdf.py

Writes data/raw/democo/FY24/annual_report.pdf - a fictional company, committed
to git, so a fresh clone can run ingest and ask end to end before anyone
downloads a 200-page annual report.

Facts are deliberately placed on KNOWN pages, which turns "are citations
correct?" into an assertion instead of a judgement call: if a question about
attrition returns anything other than page 3, page provenance is broken
somewhere in parsing or chunking.
"""

import textwrap
from pathlib import Path

import pymupdf

OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "democo" / "FY24" / "annual_report.pdf"
)

# (title, body) per page, in order. Page N below == page N in a PDF reader.
PAGES = [
    ("DemoCo Limited - Annual Report FY24",
     "This document is synthetic test data generated for pipeline verification.\n"
     "It is not a real filing and contains no real financial information.\n"
     "DemoCo Limited is a fictional company used to exercise the FilingsIQ "
     "ingestion and retrieval pipeline end to end."),

    ("Financial Highlights",
     "Total revenue for FY24 stood at Rs. 48,250 crore, compared with "
     "Rs. 41,900 crore in FY23.\n"
     "Earnings before interest and tax was Rs. 9,640 crore for the year.\n"
     "Profit after tax was Rs. 7,180 crore.\n"
     "The board recommended a final dividend of Rs. 24 per equity share."),

    ("Management Discussion and Analysis",
     "Attrition moderated significantly during the year. Voluntary attrition on "
     "a trailing twelve month basis declined to 12.4 percent in FY24 from 19.7 "
     "percent in FY23.\n"
     "The company added 8,400 employees on a net basis, taking total headcount "
     "to 214,000 as at 31 March 2024.\n"
     "Utilisation excluding trainees improved to 85.1 percent."),

    ("Technology and Generative AI",
     "The company established a dedicated generative AI practice during FY24. "
     "More than 270 client engagements involving generative AI were executed "
     "during the year.\n"
     "Approximately 61,000 employees completed foundational training in "
     "generative AI technologies.\n"
     "Management views generative AI as a durable growth driver rather than a "
     "short term cycle, while cautioning that revenue conversion remains early."),

    ("Segment Performance",
     "Banking, financial services and insurance contributed 31.2 percent of "
     "revenue in FY24.\n"
     "Retail and consumer packaged goods contributed 15.8 percent.\n"
     "Life sciences and healthcare contributed 11.4 percent.\n"
     "North America accounted for 51.3 percent of revenue, while India "
     "accounted for 6.2 percent."),

    ("Risk Factors",
     "The company faces risks from client concentration, wage inflation, "
     "currency volatility in the rupee dollar rate, and regulatory changes "
     "affecting cross border service delivery.\n"
     "Cybersecurity incidents could result in liability and reputational harm."),
]

# Layout constants for the generated page.
MARGIN_X, TITLE_Y, BODY_Y = 72, 90, 130
LINE_HEIGHT, PARA_GAP, WRAP_CHARS = 18, 6, 90


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open()
    for title, body in PAGES:
        page = document.new_page()  # A4 by default
        page.insert_text((MARGIN_X, TITLE_Y), title, fontsize=16, fontname="helv")

        y = BODY_Y
        for paragraph in body.split("\n"):
            # Wrap on WORD boundaries. A naive fixed-width slice splits words
            # across lines ("twelv" / "e month"), and because each line is a
            # separate insert_text call, extraction then yields broken tokens.
            # That would make this fixture misrepresent the parser: retrieval
            # would look worse here than on real PDFs, which wrap properly.
            for line in textwrap.wrap(paragraph, width=WRAP_CHARS) or [""]:
                page.insert_text((MARGIN_X, y), line, fontsize=11, fontname="helv")
                y += LINE_HEIGHT
            y += PARA_GAP

    document.save(OUTPUT)
    document.close()
    print(f"Wrote {OUTPUT.relative_to(Path.cwd()) if OUTPUT.is_relative_to(Path.cwd()) else OUTPUT}"
          f" ({len(PAGES)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

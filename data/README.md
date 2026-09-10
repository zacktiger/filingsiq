# Corpus layout

Source PDFs are **not** committed (see `.gitignore`): they are large, and
redistributable only from their original sources. This directory documents the
layout so the pipeline is reproducible from a fresh clone.

## Where files go

```
data/raw/{company}/{fiscal_year}/{doc_type}.pdf
```

`{doc_type}` must be one of `annual_report`, `quarterly_results`, `earnings_call`
(the closed set in `backend/app/rag/loader.py`). Company is lowercase and
fiscal year is uppercase, e.g.:

```
data/raw/tcs/FY24/annual_report.pdf
data/raw/infosys/FY24/earnings_call.pdf
data/raw/hdfcbank/FY23/quarterly_results.pdf
```

The path IS the metadata - ingestion reads company, fiscal year, and document
type straight from it, so a misplaced file is a metadata bug. `loader.py` fails
loudly on anything it cannot parse rather than skipping it.

## manifest.csv

Holds the fields a path cannot express. Optional - documents without a row still
index, just with no sector or source URL.

| Column | Meaning |
|---|---|
| `source_path` | Path relative to `data/raw/`, forward slashes. Must match exactly. |
| `company_name` | Full display name, e.g. `Tata Consultancy Services` |
| `sector` | `IT`, `Banking`, `FMCG` |
| `source_url` | The BSE / NSE / investor-relations URL the PDF came from |

Recording `source_url` matters: reviewers should be able to re-download the
exact document an answer was based on.

## Phase 1 scope

Three companies (see `project.md`). Where to get filings:

- **BSE** - bseindia.com -> Corporates -> Annual Reports
- **NSE** - nseindia.com -> Companies -> Financial Results
- **Investor relations** - usually the most reliable for earnings call transcripts

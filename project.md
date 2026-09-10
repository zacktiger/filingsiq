# FilingsIQ: Search and Q&A over Indian Company Filings

## Overview

FilingsIQ is a search engine and AI assistant for annual reports, quarterly results, and earnings call transcripts of Indian listed companies. Users can search filings with filters or ask questions in plain English and get answers with page-level citations, numeric comparisons, and charts.

The project goes beyond a basic RAG demo by focusing on retrieval quality, query understanding, structured data for numbers, and measured evaluation.

## Problem

Financial information is spread across long PDFs (often 200+ pages), mixed with tables, and inconsistent between companies. Answering a question like "compare the operating margins of three IT companies over three years" means opening nine reports and hunting through tables. Generic chatbots hallucinate numbers and cannot cite where they came from.

## Goals

- Accurate answers to factual and comparative questions, grounded in source documents
- Every answer cites the exact document and page
- Numeric questions answered from structured data, not LLM arithmetic
- A search mode with filters and highlighted snippets
- Refusal when the answer is not in the documents or the question asks for predictions or investment advice
- Measured quality, with results published in the README

## Non-goals

- Investment advice, stock recommendations, or price forecasts
- Real-time market data or live stock prices
- Coverage of all listed companies (the initial scope is deliberately small)

## Dataset scope (v1)

| Item | Scope |
|---|---|
| Companies | 10 to 15 Nifty 50 companies |
| Sectors | IT, banking, FMCG |
| Documents | Annual reports, quarterly results, earnings call transcripts |
| Time range | Last 3 financial years |
| Sources | BSE, NSE, and company investor relations pages |

## Example questions

| Category | Example |
|---|---|
| Simple lookup | What was HDFC Bank's net interest margin in FY24? |
| Numeric comparison | Compare revenue growth of TCS, Infosys, and Wipro over 3 years |
| Transcript search | What did management say about attrition in the latest call? |
| Cross-document search | Which IT company discussed GenAI most as a growth driver? |
| Should refuse | What will Infosys's revenue be next year? |

## Key features

1. **Hybrid search.** Dense and keyword (BM25) retrieval merged with Reciprocal Rank Fusion, followed by cross-encoder reranking.
2. **Query understanding.** Intent classification, company alias normalisation (HUL to Hindustan Unilever), time expression parsing (last year, Q2 FY25), and financial synonym expansion (topline to revenue).
3. **Tables to SQL.** Key financial metrics extracted into Postgres, queried through a text-to-SQL tool for numeric questions.
4. **Agentic routing.** A LangGraph agent chooses between document retrieval, SQL, or asking a clarifying question.
5. **Cited, streamed answers.** Structured output with verified citations, streamed to the UI.
6. **Search mode.** Ranked results with filters for company, sector, year, and document type.
7. **PDF viewer.** Citations open the source PDF at the cited page.
8. **Charts.** Comparison answers return data rendered as charts in the frontend.

## Architecture

### Offline indexing pipeline

1. Download filings and store raw PDFs with metadata (company, year, document type, source URL)
2. Parse text with PyMuPDF and tables with pdfplumber or Docling
3. Chunk by document structure (sections, headings), using parent-child chunks
4. Enrich chunks with metadata and extracted entities
5. Extract financial tables into Postgres
6. Embed chunks and index dense and sparse vectors in Qdrant

### Online query pipeline

1. React UI sends the query and chat history to FastAPI
2. Query understanding: intent, entities, filters, rewriting
3. Router selects RAG, SQL tool, or clarification
4. Hybrid retrieval with metadata filters, then reranking
5. Confidence check, with refusal below a tuned threshold
6. LLM generates a cited answer as structured output, streamed via SSE
7. Citations verified against retrieved context before display

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | React (Vite), Tailwind, TanStack Query, react-pdf, Recharts |
| Backend | FastAPI, Pydantic, SSE streaming |
| Orchestration | LangChain, LangGraph |
| Retrieval | bge-m3 embeddings, Qdrant, bge-reranker-v2-m3 |
| Data | Postgres (metadata, financial tables, chat history, feedback), Redis (cache, job queue) |
| Parsing | PyMuPDF, pdfplumber or Docling |
| Background jobs | Celery or RQ |
| Observability | Langfuse or LangSmith |
| Evaluation | RAGAS or LLM-as-judge, custom retrieval metrics |
| Deployment | Docker Compose, Vercel (frontend), VPS or Railway (backend), GitHub Actions |

## API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /search` | Ranked results with snippets and filters |
| `POST /chat` | Streamed answer with citations |
| `GET /documents/{id}` | Document metadata and PDF access |
| `POST /feedback` | Thumbs up or down on answers |
| `POST /ingest` | Queue new documents for indexing |

## Evaluation plan

- Build a test set of 80 to 100 questions, each with the expected answer and source page
- Split questions into the five categories listed above
- Retrieval metrics: Recall@k, MRR, nDCG
- Answer metrics: faithfulness, correctness, citation accuracy, correct refusal rate
- Track latency and cost per query
- Record a baseline, then re-run after every pipeline change
- Run the eval script in GitHub Actions on each pull request

## Milestones

| Phase | Deliverable |
|---|---|
| 1. Vertical slice | 3 companies, basic chunking, dense search, FastAPI endpoint, minimal React page |
| 2. Evaluation | Test set and baseline metrics |
| 3. Retrieval upgrades | Structure-aware chunking, hybrid search, reranking, metadata filters (each measured) |
| 4. Query understanding | Intent, aliases, time parsing, synonym expansion, query rewriting |
| 5. Structured data | Table extraction, text-to-SQL tool, LangGraph routing |
| 6. Frontend polish | Search mode, streaming chat, PDF citation viewer, charts, feedback |
| 7. Production | Docker, deployment, tracing, caching, CI evals |
| 8. Write-up | README, architecture diagram, metrics table, blog post |

## Success criteria

- Deployed live demo with a public link
- Published metrics table showing baseline versus final results per question category
- Every answer includes verifiable citations
- Correct refusals on out-of-scope and prediction questions
- README documents what was tried, what worked, and what did not

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Tables parse poorly from PDFs | Compare parsers early, and manually verify key metrics for the eval set |
| Inconsistent metric names across companies | Maintain a metric normalisation map |
| LLM miscalculates numbers | Route numeric questions to SQL, never compute in the prompt |
| Scope creep | Freeze v1 to 15 companies and 3 years until all milestones are done |
| API costs | Cache embeddings and frequent queries, and use a smaller model for routing |

## Future extensions

- Expand coverage to more companies and sectors
- Hindi and Hinglish queries
- Alerts when new filings are published
- Adapt the same pipeline to regulatory or land law documents
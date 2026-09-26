/**
 * Every number on the page, in one file, with where it came from.
 *
 * These are measured, not claimed: each traces to a run file in
 * data/eval_runs/ and is reproducible with the commands in the README. When a
 * new run changes a number, change it here and nowhere else.
 */

export const SPACE_URL = "https://huggingface.co/spaces/kshitij005/filingsiq";
export const REPO_URL = "https://github.com/zacktiger/filingsiq";

export const STACK = ["gpt-oss-120b on Groq", "bge-small-en-v1.5", "Qdrant", "Page-level citations"];

export const EXAMPLES = [
  "What was TCS's revenue in FY 2024?",
  "What was Infosys's operating margin in fiscal 2023?",
  "What was HDFC Bank's capital adequacy ratio in FY23?",
  "Who audits Infosys's financial statements?",
  "By when does TCS aim to reach net zero emissions?",
  "What will Infosys's revenue be next year?",
];

// Pages and chunks from ingest.py; source URLs from data/manifest.csv.
export const CORPUS = [
  { company: "Tata Consultancy Services", short: "TCS", sector: "IT", fy: "FY24", pages: 341, chunks: 1414, url: "https://www.bseindia.com/xml-data/corpfiling/AttachHis/d57e49b6-fd89-4fa8-b591-4f590100db1e.pdf" },
  { company: "Tata Consultancy Services", short: "TCS", sector: "IT", fy: "FY23", pages: 346, chunks: 1393, url: "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2022-23/ar/annual-report-2022-2023.pdf" },
  { company: "Infosys", short: "Infosys", sector: "IT", fy: "FY24", pages: 353, chunks: 1489, url: "https://www.bseindia.com/xml-data/corpfiling/AttachHis/25cc7fbc-154b-4463-928d-c95f2d8c5c9c.pdf" },
  { company: "Infosys", short: "Infosys", sector: "IT", fy: "FY23", pages: 358, chunks: 1547, url: "https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-23.pdf" },
  { company: "HDFC Bank", short: "HDFC Bank", sector: "Banking", fy: "FY24", pages: 585, chunks: 2186, url: "https://www.hdfc.bank.in/content/dam/hdfcbankpws/in/en/pdf/annual-reports/2023-24/integrated-annual-report-2023-24.pdf" },
  { company: "HDFC Bank", short: "HDFC Bank", sector: "Banking", fy: "FY23", pages: 447, chunks: 1795, url: "https://www.hdfc.bank.in/content/dam/hdfcbankpws/in/en/pdf/annual-reports/2022-23/reports/integrated-annual-report-2022-23.pdf" },
];

// data/eval_runs/20260925T124838Z.json (baseline) and 20260925T133950Z.json
// (fiscal-year filter), bge-small, 109 answerable questions.
export const RETRIEVAL = [
  { k: "Recall@5", before: 47.7, after: 56.0 },
  { k: "Recall@10", before: 57.8, after: 67.9 },
  { k: "Recall@20", before: 70.6, after: 76.1 },
];

// data/eval_runs/answers-20260925T134808Z, Groq gpt-oss-120b, all 121 questions.
export const ANSWERS = [
  { label: "Correct refusals", value: "83%", note: "10 of 12 forecast, advice and out-of-corpus questions declined" },
  { label: "Arithmetic violations", value: "0", note: "never computed a growth rate or ratio it was told not to" },
  { label: "Answers citing a gold page", value: "63%", note: "of 68 answered, cited a page that states the fact" },
  { label: "Correct answers", value: "42%", note: "46 of 109 answerable questions, scored on exact figures" },
];

// Of the 109 answerable questions: why each one that failed, failed.
export const DIAGNOSIS = [
  { label: "Correct", n: 46, tone: "accent" },
  { label: "Retrieval failure — right page never found", n: 49, tone: "ink" },
  { label: "Generation failure — page found, answer wrong", n: 14, tone: "warn" },
];

export const QUESTION_SET = { total: 121, answerable: 109, refusal: 12, goldPages: 441 };

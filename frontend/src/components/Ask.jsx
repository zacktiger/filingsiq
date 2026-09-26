import { useState } from "react";
import { ask } from "../api.js";
import { EXAMPLES } from "../data.js";

/**
 * Render the model's answer: **bold** becomes <strong>, and source markers -
 * which gpt-oss writes as 【S1】 as often as [S1] - become one consistent chip.
 * Deliberately not a Markdown library: these two patterns are all the answer
 * prompt produces.
 */
function AnswerText({ text }) {
  const parts = text.split(/(\*\*[^*]+\*\*|[【[]S\s*\d+(?:\s*,\s*S?\s*\d+)*[】\]])/g);
  return (
    <p className="whitespace-pre-wrap text-[17px] leading-relaxed">
      {parts.map((part, i) => {
        if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (/^[【[]S/.test(part)) {
          const nums = part.match(/\d+/g).join(", ");
          return (
            <sup key={i} className="mx-0.5 rounded bg-accent-soft px-1 font-mono text-[10px] text-accent">
              S{nums}
            </sup>
          );
        }
        return part;
      })}
    </p>
  );
}

// Company slugs as stored in the index -> the names a reader expects.
const COMPANY_NAMES = { tcs: "TCS", infosys: "Infosys", hdfcbank: "HDFC Bank" };

function Citation({ citation }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-lg border border-line bg-card">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-3 px-4 py-3 text-left">
        <span className="shrink-0 whitespace-nowrap rounded bg-accent-soft px-2 py-0.5 font-mono text-[11px] font-medium text-accent">
          p. {citation.page}
        </span>
        <span className="whitespace-nowrap text-sm font-medium">
          {COMPANY_NAMES[citation.company] ?? citation.company}
        </span>
        <span className="whitespace-nowrap font-mono text-xs text-muted">
          {citation.fiscal_year}
          <span className="hidden sm:inline"> · {citation.doc_type.replace("_", " ")}</span>
        </span>
        <span className="ml-auto whitespace-nowrap text-xs text-muted">
          {open ? "Hide" : "Show"}<span className="hidden sm:inline"> passage</span>
        </span>
      </button>
      {/* The retrieved passage itself, so a reader can check the claim without
          opening a 500-page PDF. */}
      {open && (
        <blockquote className="max-h-60 overflow-y-auto border-t border-line px-4 py-3 text-sm leading-relaxed text-muted">
          {/* PDF extraction keeps the layout's line breaks - one per table
              cell - so whitespace is collapsed to read as a passage. */}
          {citation.quote.replace(/\s+/g, " ").trim()}
        </blockquote>
      )}
    </li>
  );
}

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [asked, setAsked] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function submit(q) {
    const text = q.trim();
    if (!text || loading) return;
    setQuestion(text);
    setLoading(true);
    setError(null);
    setResult(null);
    setAsked(text);
    try {
      setResult(await ask(text));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <h1 className="font-serif text-4xl font-semibold tracking-tight">Ask the filings</h1>
      <p className="mt-2 text-sm text-muted">
        TCS, Infosys and HDFC Bank · FY23 and FY24 annual reports. Name the year for the best results.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(question);
        }}
        className="mt-6 flex gap-2"
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What was HDFC Bank's net interest margin in FY24?"
          aria-label="Your question"
          className="flex-1 rounded-lg border border-line bg-card px-4 py-3 text-sm outline-none focus:border-ink"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="rounded-lg bg-ink px-5 py-3 text-sm font-medium text-paper hover:bg-ink/85 disabled:opacity-40"
        >
          {loading ? "Reading…" : "Ask"}
        </button>
      </form>

      <div className="mt-4 flex flex-wrap gap-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => submit(q)}
            disabled={loading}
            className="rounded-full border border-line bg-card px-3 py-1.5 text-xs text-muted hover:border-ink hover:text-ink disabled:opacity-40"
          >
            {q}
          </button>
        ))}
      </div>

      {/* The model reasons before answering and a sleeping Space must wake, so
          a few seconds of silence is normal and is said so. */}
      {loading && (
        <div className="mt-10 rounded-xl border border-line bg-card p-6">
          <p className="font-mono text-xs text-muted">{asked}</p>
          <p className="mt-3 animate-pulse text-sm text-muted">
            Searching 9,830 passages and reading the closest five… the first question can take ~30s while the
            backend wakes.
          </p>
        </div>
      )}

      {error && (
        <div className="mt-10 rounded-xl border border-warn/30 bg-warn-soft p-6 text-sm text-warn">
          <p className="font-medium">Could not get an answer</p>
          <p className="mt-1">{error}</p>
        </div>
      )}

      {result && (
        <article className="mt-10">
          <p className="font-mono text-xs text-muted">{asked}</p>
          <div
            className={`mt-3 rounded-xl border p-6 ${
              result.refused ? "border-line bg-paper" : "border-line bg-card"
            }`}
          >
            {result.refused && (
              <p className="mb-2 font-mono text-[11px] uppercase tracking-widest text-muted">Declined</p>
            )}
            <AnswerText text={result.answer} />
            {!result.refused && result.citations.length === 0 && (
              <p className="mt-4 text-sm text-warn">
                This answer carried no verifiable citation. Treat it as unreliable.
              </p>
            )}
          </div>

          {result.citations.length > 0 && (
            <>
              <p className="mt-6 mb-3 font-mono text-[11px] uppercase tracking-widest text-muted">
                Sources · verified against what was retrieved
              </p>
              <ul className="space-y-2">
                {result.citations.map((c, i) => (
                  <Citation key={i} citation={c} />
                ))}
              </ul>
            </>
          )}
        </article>
      )}
    </div>
  );
}

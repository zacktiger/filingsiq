import { ANSWERS, DIAGNOSIS, QUESTION_SET, RETRIEVAL, STACK } from "../data.js";

const FEATURES = [
  {
    title: "Every claim has a page",
    body: "Each PDF page is chunked on its own, so every passage has exactly one page to cite. Citations the model invents are dropped.",
  },
  {
    title: "Knows which year you mean",
    body: "Two annual reports read almost identically. The question's fiscal year is parsed and applied as a filter before search.",
  },
  {
    title: "Refuses instead of guessing",
    body: "Forecasts, investment advice and facts outside the filings get an explicit refusal, not a plausible-sounding answer.",
  },
  {
    title: "Measured, not claimed",
    body: `A ${QUESTION_SET.total}-question set written by reading the filings, with ${QUESTION_SET.goldPages} hand-verified gold pages.`,
  },
];

function Label({ children }) {
  return <p className="mb-2 font-mono text-[11px] uppercase tracking-widest text-muted">{children}</p>;
}

export default function Landing({ onAsk }) {
  const totalAnswerable = DIAGNOSIS.reduce((sum, d) => sum + d.n, 0);
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6">
      {/* Hero */}
      <section className="flex flex-col items-center py-16 text-center sm:py-24">
        <div className="mb-6 flex flex-wrap justify-center gap-2">
          {STACK.map((item) => (
            <span key={item} className="rounded-full border border-line bg-card px-3 py-1 font-mono text-[11px] text-muted">
              {item}
            </span>
          ))}
        </div>
        <h1 className="max-w-3xl font-serif text-4xl font-semibold leading-[1.1] tracking-tight sm:text-6xl">
          Ask an annual report. Get the page it came from.
        </h1>
        <p className="mt-6 max-w-xl text-base leading-relaxed text-muted">
          Question answering over the FY23 and FY24 annual reports of TCS, Infosys and HDFC Bank —
          2,430 pages — with a citation you can check on every answer.
        </p>
        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <button onClick={onAsk} className="rounded-lg bg-ink px-6 py-3 text-sm font-medium text-paper hover:bg-ink/85">
            Ask the filings →
          </button>
          <a href="#results" className="rounded-lg border border-line bg-card px-6 py-3 text-sm font-medium hover:border-ink">
            See the measured results
          </a>
        </div>
      </section>

      {/* Features */}
      <section className="grid gap-4 border-t border-line py-12 sm:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map((f) => (
          <div key={f.title} className="rounded-xl border border-line bg-card p-5">
            <h3 className="font-serif text-lg font-semibold">{f.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
          </div>
        ))}
      </section>

      {/* Results */}
      <section id="results" className="scroll-mt-20 border-t border-line py-12">
        <Label>Evaluation · reproducible from data/eval_runs/</Label>
        <h2 className="font-serif text-3xl font-semibold tracking-tight">Measured, not claimed</h2>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">
          Numbers from a {QUESTION_SET.total}-question set written by reading the filings, not by asking
          the system what it could answer. Retrieval is scored without spending a single model token, so
          every change is measured before it ships.
        </p>

        <div className="mt-8 grid gap-6 lg:grid-cols-5">
          {/* Retrieval before/after */}
          <div className="rounded-xl border border-line bg-card p-6 lg:col-span-3">
            <div className="flex items-baseline justify-between gap-4">
              <h3 className="font-serif text-xl font-semibold">Fiscal-year filtering</h3>
              <span className="rounded-full bg-accent px-3 py-1 font-mono text-[11px] text-paper">+8.3 pts Recall@5</span>
            </div>
            <p className="mt-2 text-sm text-muted">
              Adding FY23 reports made dense search confuse consecutive years. Parsing the year from the
              question and filtering on it recovered most of the loss.
            </p>
            <table className="mt-6 w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left font-mono text-[11px] uppercase tracking-wider text-muted">
                  <th className="pb-2 font-normal">Metric</th>
                  <th className="pb-2 text-right font-normal">No filter</th>
                  <th className="pb-2 text-right font-normal text-ink">Year filter</th>
                </tr>
              </thead>
              <tbody>
                {RETRIEVAL.map((r) => (
                  <tr key={r.k} className="border-b border-line last:border-0">
                    <td className="py-3 font-mono text-xs">{r.k}</td>
                    <td className="py-3 text-right text-muted">{r.before.toFixed(1)}%</td>
                    <td className="py-3 text-right font-semibold">{r.after.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 font-mono text-[11px] text-muted">
              {QUESTION_SET.answerable} answerable questions · bge-small · top-k from Qdrant
            </p>
          </div>

          {/* Where failures come from */}
          <div className="rounded-xl border border-line bg-card p-6 lg:col-span-2">
            <h3 className="font-serif text-xl font-semibold">Where answers fail</h3>
            <p className="mt-2 text-sm text-muted">
              Of {totalAnswerable} answerable questions — the case for spending effort on retrieval, not on a
              bigger model. Gemini matched Groq on 16 of 17 questions.
            </p>
            <div className="mt-6 flex h-3 overflow-hidden rounded-full">
              {DIAGNOSIS.map((d) => (
                <div
                  key={d.label}
                  style={{ width: `${(d.n / totalAnswerable) * 100}%` }}
                  className={d.tone === "accent" ? "bg-accent" : d.tone === "warn" ? "bg-warn" : "bg-ink"}
                />
              ))}
            </div>
            <ul className="mt-4 space-y-2 text-sm">
              {DIAGNOSIS.map((d) => (
                <li key={d.label} className="flex items-start gap-3">
                  <span
                    className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${
                      d.tone === "accent" ? "bg-accent" : d.tone === "warn" ? "bg-warn" : "bg-ink"
                    }`}
                  />
                  <span className="flex-1 text-muted">{d.label}</span>
                  <span className="font-mono">{d.n}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Answer metrics */}
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {ANSWERS.map((a) => (
            <div key={a.label} className="rounded-xl border border-line bg-card p-5">
              <p className="font-serif text-4xl font-semibold">{a.value}</p>
              <p className="mt-1 text-sm font-medium">{a.label}</p>
              <p className="mt-1 text-xs leading-relaxed text-muted">{a.note}</p>
            </div>
          ))}
        </div>

        <p className="mt-6 rounded-lg border border-line bg-warn-soft p-4 text-sm leading-relaxed text-warn">
          Known limitation: figures inside complex tables can be extracted in scrambled column order, and
          the model may then quote a neighbouring year's number. Always check the cited page.
        </p>
      </section>

      {/* How it works */}
      <section className="border-t border-line py-12">
        <Label>Pipeline</Label>
        <h2 className="font-serif text-3xl font-semibold tracking-tight">How an answer is built</h2>
        <ol className="mt-8 grid gap-4 md:grid-cols-4">
          {[
            ["Parse", "PyMuPDF extracts each page separately; the page number travels with every chunk."],
            ["Understand", "The question's fiscal year (FY24, fiscal 2023, March 31, 2024…) becomes a filter."],
            ["Retrieve", "bge-small embeds the question locally; Qdrant returns the closest five passages."],
            ["Answer & verify", "The model answers only from those passages; citations are checked against them."],
          ].map(([title, body], i) => (
            <li key={title} className="rounded-xl border border-line bg-card p-5">
              <span className="font-mono text-xs text-muted">0{i + 1}</span>
              <h3 className="mt-2 font-serif text-lg font-semibold">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}

import { CORPUS } from "../data.js";

export default function Corpus() {
  const pages = CORPUS.reduce((s, d) => s + d.pages, 0);
  const chunks = CORPUS.reduce((s, d) => s + d.chunks, 0);
  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6">
      <h1 className="font-serif text-4xl font-semibold tracking-tight">What's indexed</h1>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
        Six integrated annual reports, {pages.toLocaleString("en-IN")} pages split into{" "}
        {chunks.toLocaleString("en-IN")} passages. Every passage belongs to exactly one page, which is what
        makes a page-level citation possible. Anything not in these documents gets a refusal.
      </p>

      <div className="mt-8 overflow-x-auto rounded-xl border border-line bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-line text-left font-mono text-[11px] uppercase tracking-wider text-muted">
              <th className="px-4 py-3 font-normal">Company</th>
              <th className="px-4 py-3 font-normal">Sector</th>
              <th className="px-4 py-3 font-normal">Year</th>
              <th className="px-4 py-3 text-right font-normal">Pages</th>
              <th className="px-4 py-3 text-right font-normal">Passages</th>
              <th className="px-4 py-3 font-normal">Source</th>
            </tr>
          </thead>
          <tbody>
            {CORPUS.map((d) => (
              <tr key={d.short + d.fy} className="border-b border-line last:border-0">
                <td className="px-4 py-3 font-medium">{d.company}</td>
                <td className="px-4 py-3 text-muted">{d.sector}</td>
                <td className="px-4 py-3 font-mono text-xs">{d.fy}</td>
                <td className="px-4 py-3 text-right font-mono text-xs">{d.pages}</td>
                <td className="px-4 py-3 text-right font-mono text-xs">{d.chunks.toLocaleString("en-IN")}</td>
                <td className="px-4 py-3">
                  <a href={d.url} target="_blank" rel="noreferrer" className="text-accent underline-offset-2 hover:underline">
                    PDF ↗
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-6 text-xs leading-relaxed text-muted">
        Page numbers are PDF page indices, as a PDF viewer shows them — the BSE copies carry a cover letter, so
        they can differ from the numbers printed inside a report.
      </p>
    </div>
  );
}

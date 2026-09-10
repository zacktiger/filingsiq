import { useState } from "react";

/**
 * FilingsIQ - Phase 1 UI.
 *
 * Deliberately one file with plain fetch and useState. project.md schedules
 * TanStack Query, Tailwind, react-pdf and Recharts for Phase 6; adding them now
 * would mean more code to read while the thing worth checking is whether the
 * answer and its citations are correct.
 *
 * The citation list is the point of this page. An answer without sources is a
 * failure of the whole project, so it is shown as a warning rather than
 * quietly rendering a bare paragraph.
 */
export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        // The backend returns a useful `detail` for expected failures - a
        // missing index, or one built with a different embedding model. Surface
        // it verbatim instead of a generic "something went wrong".
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${response.status})`);
      }

      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <header>
        <h1>FilingsIQ</h1>
        <p className="tagline">
          Questions about Indian company filings, answered with page-level
          citations.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="ask">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What did management say about attrition in FY24?"
          aria-label="Your question"
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      {/* Loading is called out because the model thinks before answering, so a
          few seconds of silence is normal and should not look like a hang. */}
      {loading && <p className="hint">Retrieving passages and reading them…</p>}

      {error && (
        <section className="error">
          <strong>Error</strong>
          <p>{error}</p>
        </section>
      )}

      {result && (
        <section className={result.refused ? "answer refused" : "answer"}>
          <h2>{result.refused ? "Not answered" : "Answer"}</h2>
          {/* whiteSpace: pre-wrap keeps the model's paragraph breaks without
              pulling in a markdown renderer. */}
          <p style={{ whiteSpace: "pre-wrap" }}>{result.answer}</p>

          {result.citations.length > 0 && (
            <>
              <h3>Sources</h3>
              <ol className="citations">
                {result.citations.map((citation, index) => (
                  <li key={index}>
                    <div className="cite-head">
                      <strong>{citation.company}</strong> · {citation.fiscal_year}{" "}
                      · {citation.doc_type} · <em>page {citation.page}</em>
                    </div>
                    {/* The quoted chunk is shown so a reader can check the
                        claim against the source without opening the PDF.
                        Phase 6 turns this into a link that opens the PDF at
                        the cited page. */}
                    <blockquote>{citation.quote}</blockquote>
                  </li>
                ))}
              </ol>
            </>
          )}

          {!result.refused && result.citations.length === 0 && (
            <p className="warn">
              This answer carried no verifiable citations, which should not
              happen. Treat it as unreliable.
            </p>
          )}
        </section>
      )}
    </main>
  );
}

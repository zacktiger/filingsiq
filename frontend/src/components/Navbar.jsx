import { REPO_URL } from "../data.js";

const TABS = [
  { id: "home", label: "Overview" },
  { id: "ask", label: "Ask" },
  { id: "corpus", label: "Corpus" },
];

// A free Space sleeps when idle and takes ~30s to wake, so the status is shown
// rather than letting the first question look like a hang.
const STATUS = {
  checking: { dot: "bg-muted animate-pulse", text: "Connecting" },
  online: { dot: "bg-accent", text: "Backend online" },
  offline: { dot: "bg-warn", text: "Backend asleep" },
};

export default function Navbar({ tab, setTab, status, onRetry }) {
  const s = STATUS[status];
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-paper/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3 sm:gap-6 sm:px-6">
        <button onClick={() => setTab("home")} className="flex items-baseline gap-2">
          <span className="font-serif text-xl font-semibold tracking-tight">FilingsIQ</span>
          <span className="hidden font-mono text-[10px] uppercase tracking-widest text-muted sm:inline">
            Indian filings · cited
          </span>
        </button>

        <nav className="flex gap-0.5 sm:gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`rounded-md px-2 py-1.5 text-sm transition-colors sm:px-3 ${
                tab === t.id ? "bg-ink text-paper" : "text-muted hover:text-ink"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-4">
          <button
            onClick={onRetry}
            title="Check the backend again"
            className="flex items-center gap-2 font-mono text-[11px] text-muted hover:text-ink"
          >
            <span className={`h-2 w-2 rounded-full ${s.dot}`} />
            <span className="hidden md:inline">{s.text}</span>
          </button>
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="hidden text-sm text-muted hover:text-ink sm:inline">
            GitHub
          </a>
        </div>
      </div>
    </header>
  );
}

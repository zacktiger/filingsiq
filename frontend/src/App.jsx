import { useCallback, useEffect, useState } from "react";
import { checkBackend } from "./api.js";
import { SPACE_URL } from "./data.js";
import Ask from "./components/Ask.jsx";
import Corpus from "./components/Corpus.jsx";
import Landing from "./components/Landing.jsx";
import Navbar from "./components/Navbar.jsx";

/**
 * FilingsIQ UI. Three views switched by state rather than a router: there is
 * nothing to deep-link yet, and a router is one more thing to read while the
 * point of the page is whether an answer and its citation are right.
 */
export default function App() {
  const [tab, setTab] = useState("home");
  const [status, setStatus] = useState("checking");

  const refresh = useCallback(async () => {
    setStatus("checking");
    setStatus(await checkBackend());
  }, []);

  // Checking on load also starts waking a sleeping Space, so by the time a
  // visitor has read the landing page the backend is usually ready.
  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="flex min-h-screen flex-col font-sans">
      <Navbar tab={tab} setTab={setTab} status={status} onRetry={refresh} />
      <main className="flex-1">
        {tab === "home" && <Landing onAsk={() => setTab("ask")} />}
        {tab === "ask" && <Ask />}
        {tab === "corpus" && <Corpus />}
      </main>
      <footer className="border-t border-line py-6">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-4 font-mono text-[11px] text-muted sm:px-6">
          <span>FilingsIQ · not investment advice · answers can be wrong, check the cited page</span>
          <a href={SPACE_URL} target="_blank" rel="noreferrer" className="hover:text-ink">
            Backend on Hugging Face ↗
          </a>
        </div>
      </footer>
    </div>
  );
}

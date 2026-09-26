/**
 * The one place the UI talks to a backend.
 *
 * Hosted build (VITE_HF_SPACE=kshitij005/filingsiq): calls the Gradio Space's
 * /chat endpoint through @gradio/client. Local development (unset): calls
 * FastAPI through the Vite /api proxy. Both return the same
 * {answer, citations, refused} shape, so no component knows which it is.
 */

// Trimmed of stray whitespace, dots and slashes: a value pasted as
// "kshitij005/filingsiq." (full stop and all) made the Space lookup 401.
export const HF_SPACE = import.meta.env.VITE_HF_SPACE?.trim().replace(/^[\s./]+|[\s./]+$/g, "");

let clientPromise = null;

function spaceClient() {
  // One connection per page load: connecting fetches the Space's API schema,
  // and a sleeping free Space takes a while to wake on the first one.
  clientPromise ??= import("@gradio/client").then(({ Client }) =>
    Client.connect(HF_SPACE)
  );
  return clientPromise;
}

export async function ask(question) {
  if (HF_SPACE) {
    try {
      const client = await spaceClient();
      const result = await client.predict("/chat", { question });
      return result.data[0];
    } catch (err) {
      clientPromise = null; // allow a clean retry after a failed wake-up
      // gr.Error messages (rate limit, missing key) arrive in err.message.
      throw new Error(
        err?.message || "The backend did not respond. It may be waking up — try again."
      );
    }
  }

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    // FastAPI returns a useful `detail` for expected failures (missing index,
    // wrong embedding model). Surface it verbatim.
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

/** "online" | "offline" - connecting doubles as the health check. */
export async function checkBackend() {
  try {
    if (HF_SPACE) {
      await spaceClient();
      return "online";
    }
    const response = await fetch("/api/health");
    return response.ok ? "online" : "offline";
  } catch {
    clientPromise = null;
    return "offline";
  }
}

"use client";

import { useEffect } from "react";

import { reportRenderError } from "@/lib/monitoring";

/**
 * The app shell itself failed. This renders its own document, outside every
 * provider, so it has no translations to read and no global styles to use
 * (Next.js global-error): its three words and its colours are inline, in
 * English. Everything else on screen comes from the translation files
 * (PLAN.md 2.3b).
 */
export default function GlobalError({ error, retry }: { error: Error & { digest?: string }; retry: () => void }) {
  useEffect(() => {
    reportRenderError(error);
  }, [error]);

  return (
    <html lang="en" dir="ltr">
      <body style={{ margin: 0, minHeight: "100vh", background: "#0c1012", color: "#d5dde1", fontFamily: "system-ui, sans-serif", padding: 24 }}>
        <title>FSE/ML error</title>{/* text-ok: outside every provider; see the note above */}
        <h1 style={{ color: "#d9a54a", fontSize: 16 }}>FSE/ML hit an error</h1>{/* text-ok: as above */}
        <p>Reload the page to continue.</p>{/* text-ok: as above */}
        <button type="button" onClick={() => retry()} style={{ background: "transparent", color: "#62b6cb", border: "1px solid #62b6cb", padding: "6px 10px" }}>
          Try again{/* text-ok: as above */}
        </button>
      </body>
    </html>
  );
}

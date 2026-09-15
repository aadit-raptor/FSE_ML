"use client";

import { useEffect } from "react";

import { reportRenderError } from "@/lib/monitoring";

// The app shell itself failed. Renders its own document without global styles
// (Next.js global-error), so colours are inline.
export default function GlobalError({ error, retry }: { error: Error & { digest?: string }; retry: () => void }) {
  useEffect(() => {
    reportRenderError(error);
  }, [error]);

  return (
    <html lang="en">
      <body style={{ margin: 0, minHeight: "100vh", background: "#0c1012", color: "#d5dde1", fontFamily: "system-ui, sans-serif", padding: 24 }}>
        <title>FSE/ML error</title>
        <h1 style={{ color: "#d9a54a", fontSize: 16 }}>FSE/ML hit an error</h1>
        <p>Reload the page to continue.</p>
        <button type="button" onClick={() => retry()} style={{ background: "transparent", color: "#62b6cb", border: "1px solid #62b6cb", padding: "6px 10px" }}>
          Try again
        </button>
      </body>
    </html>
  );
}

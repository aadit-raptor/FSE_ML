"use client";

import { useState } from "react";

/** Secondary action that runs an async download and reports failure inline. */
export function DownloadButton({ label = "Excel", onDownload, title }: { label?: string; onDownload: () => Promise<void>; title?: string }) {
  const [state, setState] = useState<"idle" | "busy" | "error">("idle");
  const [error, setError] = useState<string>();
  return (
    <span className="flex items-center gap-2">
      {state === "error" && (
        <span role="alert" className="font-mono text-[10px] text-loss">
          {error}
        </span>
      )}
      <button
        type="button"
        title={title ?? "Download as an Excel workbook"}
        disabled={state === "busy"}
        onClick={async () => {
          setState("busy");
          try {
            await onDownload();
            setState("idle");
          } catch (e) {
            setError((e as Error).message);
            setState("error");
          }
        }}
        className="type-action-secondary px-2 py-0.5 text-[8.5px] text-muted shadow-[inset_0_0_0_1px_#2a343a] hover:text-ink disabled:opacity-50"
      >
        {state === "busy" ? "Preparing" : `↓ ${label}`}
      </button>
    </span>
  );
}

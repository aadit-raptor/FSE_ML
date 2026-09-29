"use client";

import { useTranslations } from "next-intl";
import { useState } from "react";

/** Secondary action that runs an async download and reports failure inline. */
export function DownloadButton({ label, onDownload, title }: { label?: string; onDownload: () => Promise<void>; title?: string }) {
  const t = useTranslations("ui");
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
        title={title ?? t("downloadTitle")}
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
        {state === "busy" ? t("downloadPreparing") : t("downloadPrefix", { label: label ?? t("download") })}
      </button>
    </span>
  );
}

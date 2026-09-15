"use client";

import { useEffect } from "react";

import { reportRenderError } from "@/lib/monitoring";

// A screen that fails to render: report it (lib/monitoring.ts) and keep the
// shell usable. The error's message is not shown; it can quote deal values.
export default function ScreenError({ error, retry }: { error: Error & { digest?: string }; retry: () => void }) {
  useEffect(() => {
    reportRenderError(error);
  }, [error]);

  return (
    <div className="grid content-start gap-3 p-6" role="alert">
      <h1 className="type-alert text-[14px]">This screen hit an error</h1>
      <p className="type-body">
        Your inputs are kept. Try again, or switch screens with the menu above.
      </p>
      <button type="button" onClick={() => retry()} className="type-action-secondary justify-self-start px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
        Try again
      </button>
    </div>
  );
}

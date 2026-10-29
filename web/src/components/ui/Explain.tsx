"use client";

import { useTranslations } from "next-intl";
import { useId, useState, type ReactNode } from "react";

/**
 * Plain-English explanations (PLAN.md 6.4). Each is a template from `web/messages/<language>.json`, namespace
 * `explain`, filled with **the strings the tile itself shows**: the caller formats a figure once and hands the same
 * text to the tile and to the explanation, so the two can't differ in rounding or digit grouping. Nothing is sent
 * anywhere; the words are the catalogue's and the figures are already on screen.
 *
 * A template marks each figure `<f>{figure}</f>`; `quote` draws it, and `data-quote` is what the browser test checks
 * against the screen (e2e/explain.spec.ts). The catalogue test holds every template to no digit of its own and one
 * argument per quote (tests/test_explain.py).
 */
export const quote = (chunks: ReactNode) => (
  <span data-quote className="font-mono text-ink">
    {chunks}
  </span>
);

/** The tile's "Explain" button and the explanation it opens; the tile places the two (components/ui/Tile.tsx). */
export function useExplanation(id: string | undefined, lines: ReactNode[] | undefined) {
  const t = useTranslations("ui");
  const [open, setOpen] = useState(false);
  const panel = useId();
  if (!lines?.length) return { button: null, panel: null };
  return {
    button: (
      <button
        type="button"
        aria-expanded={open}
        aria-controls={open ? panel : undefined}
        onClick={() => setOpen(!open)}
        className={`type-action-secondary px-1.5 py-0.5 text-[8.5px] ${open ? "bg-raised text-bright" : "text-muted hover:text-ink"}`}
      >
        {open ? t("explainHide") : t("explain")}
      </button>
    ),
    panel: open ? (
      <div id={panel} role="note" data-explanation={id} className="grid gap-1.5 border-s-2 border-accent bg-panel px-2.5 py-2">
        {lines.map((line, i) => (
          <p key={i} className="type-body text-[10.5px] leading-[1.55]">
            {line}
          </p>
        ))}
      </div>
    ) : null,
  };
}

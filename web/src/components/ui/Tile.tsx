"use client";

import type { ReactNode } from "react";

import { useExplanation } from "./Explain";
import { Term } from "./Term";

const SPAN: Record<number, string> = {
  2: "col-span-2",
  3: "col-span-3",
  4: "col-span-4",
  5: "col-span-5",
  6: "col-span-6",
  7: "col-span-7",
  8: "col-span-8",
  12: "col-span-12",
};

/** Hairline-separated grid of result tiles (12 columns). */
export function Tiles({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-12 content-start gap-px bg-line">{children}</div>;
}

export function Tile({
  span,
  title,
  unit,
  children,
  aside,
  action,
  term,
  unitTerm,
  explain,
  data,
}: {
  span: keyof typeof SPAN;
  title: string;
  unit?: string;
  children: ReactNode;
  aside?: ReactNode;
  /** Shown after the unit, e.g. a download button */
  action?: ReactNode;
  /** The title's glossary entry (PLAN.md 6.4) */
  term?: string | null;
  /** The unit's, e.g. the money unit: crores, lakhs */
  unitTerm?: string | null;
  /** The tile's plain-English explanation, one paragraph each (./Explain.tsx); an "Explain" button opens it */
  explain?: ReactNode[];
  /** `data-*` attributes for the section, e.g. { "data-facility-tile": "" } */
  data?: Record<`data-${string}`, string>;
}) {
  const explanation = useExplanation(title, explain);
  return (
    <section className={`${SPAN[span]} grid min-w-0 content-start gap-2 bg-canvas px-3 py-2.5`} aria-label={title} {...data}>
      <header className="flex min-h-4 items-center justify-between gap-2">
        <h2 className="type-result-title">
          <Term id={term}>{title}</Term>
        </h2>
        <span className="flex items-center gap-2.5">
          {aside ??
            (unit && (
              <span className="font-mono text-[10px] text-dim">
                <Term id={unitTerm}>{unit}</Term>
              </span>
            ))}
          {action}
          {explanation.button}
        </span>
      </header>
      {explanation.panel}
      {children}
    </section>
  );
}

export function Kpi({
  title,
  value,
  sub,
  lead,
  tone,
  span = 2,
  term,
}: {
  title: string;
  value: string;
  sub?: string;
  lead?: boolean;
  tone?: "gain" | "loss" | "attention";
  span?: 2 | 3 | 4;
  /** The title's glossary entry (PLAN.md 6.4) */
  term?: string | null;
}) {
  const toneClass = tone === "gain" ? "text-gain" : tone === "loss" ? "text-loss" : tone === "attention" ? "text-attention" : "text-muted";
  return (
    <section className={`${SPAN[span]} grid min-w-0 content-start gap-2 bg-canvas px-3 py-2.5`} aria-label={title}>
      <h2 className="type-result-title min-h-4">
        <Term id={term}>{title}</Term>
      </h2>
      <p className={`type-figure text-[22px] leading-[1.1] ${lead ? "text-accent" : "text-ink"}`} data-kpi={title}>
        {value}
      </p>
      {sub && <p className={`font-mono text-[10.5px] ${toneClass}`}>{sub}</p>}
    </section>
  );
}

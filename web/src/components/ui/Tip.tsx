"use client";

import * as Tooltip from "@radix-ui/react-tooltip";
import { useTranslations } from "next-intl";
import { type ReactNode, useId, useState } from "react";

/**
 * Input tips (PLAN.md 7.1): a sentence or two on what an input means and how to choose it, from
 * `web/messages/<language>.json` under `tips`. A control points `aria-describedby` at the tip's
 * words (`TipText`, kept out of sight and outside the label, so it is the control's description and
 * never part of its name), and its label shows the same words on hover (`TipLabel`).
 *
 * The shared inputs (NumberField, CellInput, Switch, MoneySelects, FiscalSelects) take the tip's key
 * as a required prop; a control written straight into a component uses these three itself.
 * tests/test_onboarding.py and e2e/tips.spec.ts hold every control to having one.
 *
 * i18n-keys: tips.*
 */
export type TipHandle = { key: string; id: string; text: string | null };

export function useTip(key: string): TipHandle {
  const t = useTranslations("tips");
  const id = `${useId()}-tip`;
  return { key, id, text: t.has(key) ? t(key) : null };
}

/** Tips for a list of controls drawn in a loop (where a hook per control can't be called), by key. */
export function useTips(keys: readonly string[]): Record<string, TipHandle> {
  const t = useTranslations("tips");
  const base = useId();
  return Object.fromEntries(keys.map((key) => [key, { key, id: `${base}-tip-${key}`, text: t.has(key) ? t(key) : null }]));
}

/** The tip's words for screen readers. A key the catalogue lacks is marked, and the browser test finds it. */
export function TipText({ tip }: { tip: TipHandle }) {
  if (!tip.text) return <span id={tip.id} hidden data-tip-missing={tip.key} />;
  return (
    <span id={tip.id} data-tip={tip.key} className="sr-only">
      {tip.text}
    </span>
  );
}

/** A control's label, showing its tip on hover. */
export function TipLabel({ tip, children }: { tip: TipHandle; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  if (!tip.text) return <>{children}</>;
  return (
    <Tooltip.Provider delayDuration={300}>
      <Tooltip.Root open={open} onOpenChange={setOpen}>
        <Tooltip.Trigger asChild>
          <span className="cursor-help decoration-dim decoration-dotted underline-offset-2 hover:underline">{children}</span>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            data-tip-popup={tip.key}
            side="top"
            align="start"
            sideOffset={4}
            collisionPadding={8}
            className="type-body z-50 max-w-72 border border-line bg-raised px-2.5 py-2 text-[10.5px] normal-case text-ink"
          >
            {tip.text}
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

/** An `aria-describedby` value from the ids given (a tip, an error), or none. */
export function describedBy(...ids: (string | false | null | undefined)[]): string | undefined {
  const joined = ids.filter(Boolean).join(" ");
  return joined || undefined;
}

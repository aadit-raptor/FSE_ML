"use client";

import * as Tooltip from "@radix-ui/react-tooltip";
import { useTranslations } from "next-intl";
import type { ReactNode } from "react";

/**
 * A label with its glossary entry on hover and on keyboard focus (PLAN.md 6.4). The definitions live in
 * `web/messages/<language>.json` under `glossary`, so another language follows by adding its file.
 *
 * `data-term` is drawn only for an id the glossary defines: a label whose term is missing shows plainly, and the
 * browser test that every label on the explained tiles carries a term (e2e/explain.spec.ts) catches it.
 *
 * i18n-keys: glossary.*
 */
export function Term({ id, children }: { id: string | null | undefined; children: ReactNode }) {
  const t = useTranslations("glossary");
  if (!id || !t.has(id)) return <>{children}</>;
  return (
    <Tooltip.Provider delayDuration={250}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <span data-term={id} tabIndex={0} className="cursor-help underline decoration-dim decoration-dotted underline-offset-2">
            {children}
          </span>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            data-glossary={id}
            side="top"
            sideOffset={4}
            collisionPadding={8}
            className="type-body z-50 max-w-72 border border-line bg-raised px-2.5 py-2 text-[10.5px] normal-case text-ink"
          >
            {t(id)}
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

/**
 * The same for a label drawn inside a chart: SVG can't hold the pop-up, so the definition is the group's own
 * title, which the browser shows on hover.
 */
export function SvgTerm({ id, children }: { id: string | null | undefined; children: ReactNode }) {
  const t = useTranslations("glossary");
  if (!id || !t.has(id)) return <>{children}</>;
  return (
    <g data-term={id} className="cursor-help">
      <title>{t(id)}</title>
      {children}
    </g>
  );
}

"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";

import { useDeal } from "@/components/deal/DealProvider";
import { stepHref, WORKSPACES, workspaceModes } from "@/lib/nav";

import { useModes } from "./useModes";

/**
 * The launcher: where every sign-in lands, and where the brand mark leads.
 * One tile per workspace, each opening on its first mode's first step and
 * listing its modes, so any of them is one click away.
 *
 * i18n-keys: nav.*
 */
export function Launcher() {
  const t = useTranslations("launcher");
  const nav = useTranslations("nav");
  const deal = useTranslations("deal");
  const shown = useModes();
  const { current } = useDeal();

  return (
    <div className="grid content-start gap-4 p-6">
      <h1 className="type-result-title text-[18px]">{t("title")}</h1>
      <p className="type-body max-w-[640px]">{t("intro")}</p>

      <ul className="grid max-w-[960px] grid-cols-[repeat(auto-fit,minmax(min(100%,300px),1fr))] gap-px border border-line bg-line">
        {WORKSPACES.map((w) => {
          const modes = workspaceModes(w, shown);
          const first = modes[0];
          if (!first) return null;
          const name = nav(w.labelKey);
          return (
            <li key={w.slug} data-workspace={w.slug} className="grid content-start gap-3 bg-panel p-5">
              <h2 className="type-tab text-[13px] text-bright">{name}</h2>
              <p className="type-body">{nav(w.summaryKey)}</p>

              {/* The live line: what this workspace has open */}
              {w.slug === "lbo" && (
                <p className="grid gap-0.5">
                  <span className="type-control">{t("openDeal")}</span>
                  <span className="type-input-label truncate">{current?.name ?? deal("unsavedDeal")}</span>
                </p>
              )}

              <nav aria-label={t("modesOf", { workspace: name })}>
                <ul className="flex flex-wrap gap-x-4 gap-y-1">
                  {modes.map((m) => (
                    <li key={m.slug}>
                      <Link href={stepHref(m.slug, m.steps[0].slug)} className="type-step hover:text-ink">
                        {nav(m.labelKey)}
                      </Link>
                    </li>
                  ))}
                </ul>
              </nav>

              <Link
                href={stepHref(first.slug, first.steps[0].slug)}
                className="type-action-secondary justify-self-start px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]"
              >
                {t("open", { workspace: name })}
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

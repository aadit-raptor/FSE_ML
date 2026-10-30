"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";
import { useEffect, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { api } from "@/lib/api/client";
import { stepHref, WORKSPACES, workspaceModes } from "@/lib/nav";

import { useModes } from "./useModes";

/**
 * The launcher: where every sign-in lands, and where the brand mark leads.
 * One tile per workspace, each opening on its first mode's first step and
 * listing its modes, so any of them is one click away. An account with no saved deal yet is offered
 * the guided first deal (PLAN.md 7.1).
 *
 * i18n-keys: nav.*
 */
export function Launcher() {
  const t = useTranslations("launcher");
  const nav = useTranslations("nav");
  const deal = useTranslations("deal");
  const shown = useModes();
  const { current } = useDeal();
  const newHere = useNoSavedDeals();

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

              {w.slug === "lbo" && newHere && (
                <div className="grid gap-1.5 border border-accent/40 bg-canvas p-3" data-guide="first-deal">
                  <p className="type-body">{t("firstDealIntro")}</p>
                  <Link href="/deal/guided" className="type-action justify-self-start bg-accent px-3 py-1.5 text-bg">
                    {t("firstDeal")}
                  </Link>
                </div>
              )}

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

/** True once the account is known to have no saved deal (archived ones count as having one). */
function useNoSavedDeals(): boolean {
  const [none, setNone] = useState(false);
  useEffect(() => {
    let live = true;
    void api
      .GET("/api/deals", { params: { query: { archived: true } } })
      .then(({ data }) => live && setNone(!!data && data.deals.length === 0))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);
  return none;
}

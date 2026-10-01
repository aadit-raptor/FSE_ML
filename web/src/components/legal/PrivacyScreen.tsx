"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";

import { monthYear } from "@/lib/locale";

/** The policy's date, as an ISO month: change it with the words */
const UPDATED = "2026-10";
const ISSUES = "https://github.com/aadit-raptor/FSE_ML/issues";

/**
 * The privacy policy (PLAN.md 0.2). Public: Google's consent screen links to
 * it, so a signed-out visitor must be able to read it (lib/auth/mode.ts
 * PUBLIC_ROUTES). Every claim mirrors what the code does -- CLAUDE.md
 * "Accounts" and "Monitoring" -- so a change there changes this page.
 */
export function PrivacyScreen() {
  const t = useTranslations("privacy");
  const sections: [string, string[]][] = [
    [t("storedTitle"), [t("storedSignIn"), t("storedProfile"), t("storedDeals")]],
    [t("googleTitle"), [t("google")]],
    [t("processorsTitle"), [t("processors")]],
    [t("notTitle"), [t("not")]],
    [t("keepTitle"), [t("keep")]],
  ];
  return (
    <div className="h-full overflow-auto">
      <article className="mx-auto grid max-w-[680px] gap-5 px-6 py-10">
        <header className="grid gap-1.5">
          <h1 className="type-result-title text-[16px]">{t("title")}</h1>
          <p className="font-mono text-[10px] text-dim">{t("updated", { date: monthYear(UPDATED) })}</p>
          <p className="type-body">{t("intro")}</p>
        </header>
        {sections.map(([title, paragraphs]) => (
          <section key={title} className="grid gap-1.5">
            <h2 className="type-control">{title}</h2>
            {paragraphs.map((p) => (
              <p key={p} className="type-body">
                {p}
              </p>
            ))}
          </section>
        ))}
        <section className="grid gap-1.5">
          <h2 className="type-control">{t("contactTitle")}</h2>
          <p className="type-body">
            {t.rich("contact", {
              link: () => (
                <a href={ISSUES} className="text-accent underline" rel="noopener noreferrer" target="_blank">
                  {ISSUES.replace("https://", "")}
                </a>
              ),
            })}
          </p>
        </section>
        <Link href="/" className="type-action-secondary w-fit px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
          {t("back")}
        </Link>
      </article>
    </div>
  );
}

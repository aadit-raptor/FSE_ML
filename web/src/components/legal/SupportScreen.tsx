"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";

/** Where people ask for help: an https page, since Microsoft's store refuses an email address */
const ISSUES = "https://github.com/aadit-raptor/FSE_ML/issues";

/**
 * The support page: the Excel add-in's manifest names it as its `SupportUrl`,
 * and Microsoft's store listing links to it (DEPLOY.md "Listing it in
 * Microsoft's store"). Public like the privacy policy (lib/auth/mode.ts
 * PUBLIC_ROUTES): the store's reviewers and anyone deciding whether to install
 * the add-in read it signed out.
 */
export function SupportScreen() {
  const t = useTranslations("support");
  const sections: [string, string[]][] = [
    [t("whatTitle"), [t("what"), t("whatAddin")]],
    [t("accountTitle"), [t("account")]],
    [t("installTitle"), [t("install")]],
    [t("questionsTitle"), [t("questionGoogle"), t("questionEnded"), t("questionRemove")]],
  ];
  const link = (href: string, text: string) => (
    <a href={href} className="text-accent underline" rel="noopener noreferrer" target="_blank">
      {text}
    </a>
  );
  return (
    <div className="h-full overflow-auto">
      <article className="mx-auto grid max-w-[680px] gap-5 px-6 py-10">
        <header className="grid gap-1.5">
          <h1 className="type-result-title text-[16px]">{t("title")}</h1>
          <p className="type-body">{t("intro")}</p>
        </header>
        <section className="grid gap-1.5">
          <h2 className="type-control">{t("helpTitle")}</h2>
          <p className="type-body">{t.rich("help", { link: () => link(ISSUES, ISSUES.replace("https://", "")) })}</p>
          <p className="type-body">{t("helpPublic")}</p>
        </section>
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
          <h2 className="type-control">{t("privacyTitle")}</h2>
          <p className="type-body">
            {t.rich("privacy", {
              link: (chunks) => (
                <Link href="/privacy" className="text-accent underline">
                  {chunks}
                </Link>
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

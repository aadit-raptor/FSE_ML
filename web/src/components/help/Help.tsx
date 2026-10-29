"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";
import { Fragment, type ReactNode, useEffect } from "react";

import { formatDateTime } from "@/lib/locale";
import index from "@/lib/help/index.json";
import { markNewsSeen, WHATS_NEW } from "@/lib/help/whatsNew";

/**
 * In-app help (PLAN.md 7.1): the methodology's sections as articles, and what's new.
 *
 * An article is data written by `python -m ops.help_articles` from docs/methodology.md: blocks of
 * inline pieces, drawn here with the app's own elements, so no Markdown and no HTML from a file
 * reaches the page. The articles are in English, as the methodology is; the words around them are
 * the account's language.
 *
 * i18n-keys: help.summary_*
 */
export type Inline = { text: string; code?: boolean; strong?: boolean; em?: boolean; href?: string; article?: string; anchor?: string };
export type Block =
  | { type: "heading"; level: number; id: string; inlines: Inline[] }
  | { type: "paragraph"; inlines: Inline[] }
  | { type: "list"; ordered: boolean; items: Inline[][] }
  | { type: "code"; text: string }
  | { type: "table"; header: Inline[][]; rows: Inline[][][] };
export type Article = { slug: string; title: string; blocks: Block[] };

const LINK = "text-accent underline decoration-dotted underline-offset-2 hover:decoration-solid";

function HelpNav({ current }: { current?: string }) {
  const t = useTranslations("help");
  return (
    <nav aria-label={t("articles")} className="grid content-start gap-0.5 border-e border-line pe-4">
      <Link href="/help/whats-new" aria-current={current === "whats-new" ? "page" : undefined} className={`type-step py-0.5 hover:text-ink ${current === "whats-new" ? "text-accent" : ""}`}>
        {t("whatsNew")}
      </Link>
      <span className="type-control mt-3 mb-1 text-dim">{t("methodology")}</span>
      {index.articles.map((a) => (
        <Link
          key={a.slug}
          href={`/help/${a.slug}`}
          aria-current={current === a.slug ? "page" : undefined}
          className={`type-step py-0.5 hover:text-ink ${current === a.slug ? "text-accent" : ""}`}
        >
          {a.title}
        </Link>
      ))}
    </nav>
  );
}

function HelpFrame({ current, children }: { current?: string; children: ReactNode }) {
  return (
    <div className="grid h-full min-h-0 grid-cols-[260px_minmax(0,1fr)] overflow-hidden">
      <div className="overflow-y-auto p-5">
        <HelpNav current={current} />
      </div>
      <div className="overflow-y-auto p-6">{children}</div>
    </div>
  );
}

/** /help: what help holds, each article with its one-line summary. */
export function HelpIndex() {
  const t = useTranslations("help");
  return (
    <HelpFrame>
      <div className="grid max-w-[760px] content-start gap-4">
        <h1 className="type-result-title text-[18px]">{t("title")}</h1>
        <p className="type-body">{t("intro")}</p>
        <ul className="grid gap-px border border-line bg-line">
          <li className="grid gap-0.5 bg-panel px-3 py-2">
            <Link href="/help/whats-new" className="type-input-label text-accent hover:underline">
              {t("whatsNew")}
            </Link>
            <span className="type-body text-[10.5px]">{t("whatsNewSummary")}</span>
          </li>
          {index.articles.map((a) => (
            <li key={a.slug} className="grid gap-0.5 bg-panel px-3 py-2">
              <Link href={`/help/${a.slug}`} className="type-input-label text-accent hover:underline">
                {a.title}
              </Link>
              <span className="type-body text-[10.5px]">{t(`summary_${a.slug}`)}</span>
            </li>
          ))}
        </ul>
      </div>
    </HelpFrame>
  );
}

function Inlines({ parts }: { parts: Inline[] }) {
  return (
    <>
      {parts.map((p, i) => {
        const text = p.code ? <code className="bg-raised px-1 font-mono text-[0.92em] text-ink">{p.text}</code> : p.text;
        if (p.article) {
          return (
            <Link key={i} href={`/help/${p.article}${p.anchor ? `#${p.anchor}` : ""}`} className={LINK}>
              {text}
            </Link>
          );
        }
        if (p.href) {
          return (
            <a key={i} href={p.href} target="_blank" rel="noopener noreferrer" className={LINK}>
              {text}
            </a>
          );
        }
        if (p.strong) return <strong key={i} className="font-semibold text-bright">{text}</strong>;
        if (p.em) return <em key={i}>{text}</em>;
        return <Fragment key={i}>{text}</Fragment>;
      })}
    </>
  );
}

function BlockView({ block }: { block: Block }) {
  switch (block.type) {
    case "heading": {
      const className = "type-result-title mt-3 scroll-mt-4 text-[13px]";
      return block.level <= 2 ? (
        <h2 id={block.id} className={className}>
          <Inlines parts={block.inlines} />
        </h2>
      ) : (
        <h3 id={block.id} className={`${className} text-[12px]`}>
          <Inlines parts={block.inlines} />
        </h3>
      );
    }
    case "paragraph":
      return (
        <p className="type-body text-[11px] leading-relaxed text-soft">
          <Inlines parts={block.inlines} />
        </p>
      );
    case "list": {
      const items = block.items.map((item, i) => (
        <li key={i} className="type-body text-[11px] leading-relaxed text-soft">
          <Inlines parts={item} />
        </li>
      ));
      return block.ordered ? <ol className="grid list-decimal gap-1 ps-5">{items}</ol> : <ul className="grid list-disc gap-1 ps-5">{items}</ul>;
    }
    case "code":
      return <pre className="overflow-x-auto border border-line bg-canvas px-3 py-2 font-mono text-[11px] leading-relaxed text-ink">{block.text}</pre>;
    case "table":
      return (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse font-mono text-[10.5px]">
            <thead>
              <tr>
                {block.header.map((cell, i) => (
                  <th key={i} scope="col" className="type-input-label border-b border-line px-2 py-1 text-start font-normal">
                    <Inlines parts={cell} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, r) => (
                <tr key={r} className="border-b border-grid align-top">
                  {row.map((cell, i) => (
                    <td key={i} className="px-2 py-1 text-soft">
                      <Inlines parts={cell} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
  }
}

/** One article: a section of the methodology. */
export function HelpArticle({ article }: { article: Article }) {
  const t = useTranslations("help");
  const app = useTranslations("app");
  useEffect(() => {
    document.title = [article.title, t("title"), app("brand")].join(" · ");
  }, [article.title, t, app]);
  return (
    <HelpFrame current={article.slug}>
      <article className="grid max-w-[820px] content-start gap-3">
        <p className="type-control text-dim">{t("fromMethodology")}</p>
        <h1 className="type-result-title text-[18px]">{article.title}</h1>
        <p className="type-body text-[10.5px] text-muted">{t(`summary_${article.slug}`)}</p>
        {article.blocks.map((b, i) => (
          <BlockView key={i} block={b} />
        ))}
      </article>
    </HelpFrame>
  );
}

/** What's new: the releases, newest first, each linking to where it is. Seen once shown. */
export function WhatsNew() {
  const t = useTranslations("help");
  const w = useTranslations("whatsNew");
  useEffect(() => markNewsSeen(), []);
  return (
    <HelpFrame current="whats-new">
      <div className="grid max-w-[760px] content-start gap-4">
        <h1 className="type-result-title text-[18px]">{t("whatsNew")}</h1>
        <p className="type-body">{t("whatsNewIntro")}</p>
        <ol className="grid gap-px border border-line bg-line">
          {WHATS_NEW.map((e) => (
            <li key={e.id} data-whats-new={e.date} className="grid gap-1 bg-panel px-4 py-3">
              <span className="font-mono text-[10px] text-dim">
                {formatDateTime(new Date(`${e.date}T00:00:00Z`), { dateStyle: "long", timeZone: "UTC" })}
              </span>
              <h2 className="type-input-group text-[12px] text-bright">{w(`${e.id}Title`)}</h2>
              <p className="type-body text-[10.5px]">{w(`${e.id}Body`)}</p>
              <Link href={e.href} className="type-step justify-self-start text-accent hover:underline">
                {t("openIt")}
              </Link>
            </li>
          ))}
        </ol>
      </div>
    </HelpFrame>
  );
}

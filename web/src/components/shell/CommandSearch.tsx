"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useTranslations } from "next-intl";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { stepHref } from "@/lib/nav";

import { useModes } from "./useModes";

import { useWorkspace } from "./workspace";

type Entry = { href: string; mode: string; step: string; summary: string };

/** Every word of the query must appear in the mode, step or summary. */
function searchEntries(query: string, entries: Entry[]): Entry[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return entries;
  return entries.filter((e) => {
    const hay = `${e.mode} ${e.step} ${e.summary}`.toLowerCase();
    return words.every((w) => hay.includes(w));
  });
}

/** Mode, step and summary in the account's language, so search matches what is on screen. */
function useEntries(): Entry[] {
  const nav = useTranslations("nav");
  const modes = useModes();
  return useMemo(
    () =>
      modes.flatMap((m) =>
        m.steps.map((s) => ({
          href: stepHref(m.slug, s.slug),
          mode: nav(m.labelKey),
          step: nav(s.labelKey),
          summary: nav(s.summaryKey),
        })),
      ),
    [nav, modes],
  );
}

export function CommandSearch() {
  const { searchOpen, setSearchOpen } = useWorkspace();
  const t = useTranslations("shell");
  return (
    <Dialog.Root open={searchOpen} onOpenChange={setSearchOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-[rgba(5,7,8,0.6)]" />
        <Dialog.Content
          aria-describedby={undefined}
          className="fixed top-[72px] left-1/2 z-50 w-[580px] max-w-[calc(100vw-32px)] -translate-x-1/2 border border-line-strong bg-[#141b1f] shadow-[0_30px_70px_-20px_rgba(0,0,0,0.8)]"
        >
          <Dialog.Title className="sr-only">{t("searchTitle")}</Dialog.Title>
          {/* Remounted on every open, so the query starts empty */}
          <SearchBody onDone={() => setSearchOpen(false)} />
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function SearchBody({ onDone }: { onDone: () => void }) {
  const router = useRouter();
  const t = useTranslations("shell");
  const entries = useEntries();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const results = useMemo(() => searchEntries(query, entries), [query, entries]);

  const go = (entry: Entry | undefined) => {
    if (!entry) return;
    onDone();
    router.push(entry.href);
  };

  return (
    <>
      <input
        autoFocus
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setActive((i) => Math.min(i + 1, results.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((i) => Math.max(i - 1, 0));
          } else if (e.key === "Enter") {
            e.preventDefault();
            go(results[active]);
          }
        }}
        placeholder={t("searchPlaceholder")}
        aria-label={t("searchPlaceholder")}
        role="combobox"
        aria-expanded
        aria-controls="command-results"
        aria-activedescendant={results[active] ? `cmd-${active}` : undefined}
        className="type-input-label w-full border-b border-line bg-transparent px-4 py-3 text-[12px] outline-none placeholder:text-dim"
      />
      <ul id="command-results" role="listbox" className="max-h-[360px] overflow-y-auto py-1">
        {results.map((r, i) => (
          <li
            key={r.href}
            id={`cmd-${i}`}
            role="option"
            aria-selected={i === active}
            onMouseMove={() => setActive(i)}
            onClick={() => go(r)}
            className={`grid cursor-pointer grid-cols-[150px_1fr] gap-3 px-4 py-2 ${i === active ? "bg-[#1f2a30] shadow-[inset_2px_0_0_var(--color-accent)]" : ""}`}
          >
            <span className="type-control self-center">{r.mode}</span>
            <span className="grid gap-0.5">
              <span className="type-input-label">{r.step}</span>
              <span className="type-body text-[9px] leading-snug">{r.summary}</span>
            </span>
          </li>
        ))}
        {!results.length && <li className="type-body px-4 py-3">{t("searchNoResults", { query })}</li>}
      </ul>
    </>
  );
}

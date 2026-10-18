"use client";

import { useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";

import { PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { loadOffice, officePlatform, openDialog } from "@/lib/excel/office";
import { applyPass, endPass, parsePassMessage, savePass, storedPass, type AddinPass } from "@/lib/excel/session";
import { periodCells, writeCells, type Written } from "@/lib/excel/write";
import { listCoverage, type CoverageCompany } from "@/lib/research";

import { ModelInPane } from "./ModelInPane";
import { ReadInPane } from "./ReadInPane";

/**
 * The Excel add-in's task pane (PLAN.md R3, /excel): sign in through Office's dialog, then import a covered
 * company's accepted quarter as cells -- the checked workbook's own cells, subtotals as formulas -- or read a
 * new results PDF here, answer its amber figures, accept it and import it.
 *
 * Signed in, every call carries the add-in's pass (lib/excel/session.ts), never the site's session.
 */
type Office = NonNullable<Awaited<ReturnType<typeof loadOffice>>>;

export function ExcelPane() {
  const t = useTranslations("excel");
  const [office, setOffice] = useState<Office | null | undefined>(undefined);
  const [pass, setPass] = useState<AddinPass | null>(null);
  const [ended, setEnded] = useState(false);
  const [signing, setSigning] = useState<"idle" | "waiting">("idle");
  const [signInProblem, setSignInProblem] = useState<string | null>(null);

  const end = useCallback(() => {
    savePass(null);
    setPass(null);
    setEnded(true);
  }, []);
  applyPass(pass, end);

  useEffect(() => {
    let cancelled = false;
    void loadOffice().then((found) => {
      if (cancelled) return;
      setOffice(found);
      setPass(storedPass());
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = async () => {
    if (!office) return;
    setSigning("waiting");
    setSignInProblem(null);
    const platform = officePlatform(office);
    const outcome = await openDialog(office, `/excel/connect?platform=${encodeURIComponent(platform)}`);
    setSigning("idle");
    if ("message" in outcome) {
      const found = parsePassMessage(outcome.message);
      if (found) {
        savePass(found);
        setEnded(false);
        setPass(found);
      } else setSignInProblem(t("signInClosed"));
    } else if ("closed" in outcome) setSignInProblem(t("signInClosed"));
    else setSignInProblem(t("signInFailed", { code: String(outcome.failed) }));
  };

  if (office === undefined) {
    return <p className="type-step p-4" role="status">{t("loading")}</p>;
  }
  if (office === null) {
    return (
      <Pane>
        <p className="type-body" data-pane-state="outside">{t("outsideOffice")}</p>
      </Pane>
    );
  }
  if (!pass) {
    return (
      <Pane>
        <div className="grid gap-3" data-pane-state="signed-out">
          {ended && <p className="type-alert" role="alert">{t("ended")}</p>}
          <p className="type-body">{t("signInHint")}</p>
          <p className="type-body text-muted">{t("googleNote")}</p>
          <div>
            <PrimaryButton disabled={signing === "waiting"} onClick={() => void signIn()}>
              {t("signIn")}
            </PrimaryButton>
          </div>
          {signing === "waiting" && <p className="type-step" role="status">{t("signingIn")}</p>}
          {signInProblem && <p className="type-alert" role="alert">{signInProblem}</p>}
        </div>
      </Pane>
    );
  }
  return (
    <Pane>
      <div className="grid gap-4" data-pane-state="signed-in">
        <div className="flex items-center justify-between gap-2">
          <span className="type-body truncate">{pass.label ? t("signedInAs", { name: pass.label }) : t("signedIn")}</span>
          <SecondaryButton
            onClick={() => {
              void endPass().then(() => {
                setPass(null);
                setEnded(false);
              });
            }}
          >
            {t("signOut")}
          </SecondaryButton>
        </div>
        <SignedIn office={office} />
      </div>
    </Pane>
  );
}

function Pane({ children }: { children: React.ReactNode }) {
  const t = useTranslations("excel");
  return (
    <div className="grid gap-4 p-4">
      <h1 className="type-result-title text-[13px]">{t("title")}</h1>
      {children}
    </div>
  );
}

function SignedIn({ office }: { office: Office }) {
  const t = useTranslations("excel");
  const reader = useTranslations("reader");
  const [companies, setCompanies] = useState<CoverageCompany[] | null>(null);
  const [companyId, setCompanyId] = useState("");
  const [periodId, setPeriodId] = useState("");
  const [writing, setWriting] = useState(false);
  const [written, setWritten] = useState<Written | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void listCoverage().then((found) => {
      if (!cancelled) setCompanies(found?.companies ?? []);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const company = (companies ?? []).find((c) => c.id === companyId) ?? companies?.[0] ?? null;
  const periods = company?.periods ?? [];
  const period = periods.find((p) => p.id === periodId) ?? periods[0] ?? null;

  const importPeriods = async (ids: string[]) => {
    setWriting(true);
    setProblem(null);
    setWritten(null);
    try {
      const support = {
        notes: office.context.requirements.isSetSupported("ExcelApi", "1.18"),
        comments: office.context.requirements.isSetSupported("ExcelApi", "1.10"),
      };
      const names: string[] = [];
      let notes: Written["notes"] = "none";
      for (const id of ids) {
        const cells = await periodCells(id);
        if (!cells) throw new Error("cells");
        const run = window.Excel?.run;
        if (!run) throw new Error("excel");
        const done = await writeCells((batch) => run(batch), cells, support);
        names.push(...done.sheets);
        notes = done.notes;
      }
      setWritten({ sheets: names, notes });
    } catch {
      setProblem(t("writeFailed"));
    } finally {
      setWriting(false);
    }
  };

  const scopeWord = (scope: string) => reader(`scope_${scope}`);

  return (
    <>
      <section className="grid gap-2.5">
        <h2 className="type-input-group">{t("importTitle")}</h2>
        {companies !== null && companies.length === 0 && <p className="type-body">{t("noCompanies")}</p>}
        {companies !== null && companies.length > 0 && (
          <>
            <label className="grid gap-1">
              <span className="type-input-label">{t("company")}</span>
              <select
                value={company?.id ?? ""}
                onChange={(e) => {
                  setCompanyId(e.target.value);
                  setPeriodId("");
                }}
                className="border border-line bg-field px-2 py-1.5 text-[12px] text-ink"
              >
                {companies.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            </label>
            {periods.length === 0 ? (
              <p className="type-body">{t("noPeriods")}</p>
            ) : (
              <label className="grid gap-1">
                <span className="type-input-label">{t("period")}</span>
                <select
                  value={period?.id ?? ""}
                  onChange={(e) => setPeriodId(e.target.value)}
                  className="border border-line bg-field px-2 py-1.5 text-[12px] text-ink"
                >
                  {periods.map((p) => (
                    <option key={p.id} value={p.id}>
                      {t("periodOption", { end: p.period_end, months: String(p.months), scope: scopeWord(p.scope) })}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <div>
              <PrimaryButton disabled={!period || writing} onClick={() => period && void importPeriods([period.id])}>
                {t("import")}
              </PrimaryButton>
            </div>
          </>
        )}
        {writing && <p className="type-step" role="status">{t("importing")}</p>}
        {written && (
          <div className="grid gap-1" data-written={written.sheets.length}>
            <p className="type-body" role="status">
              {t("written", { count: written.sheets.length, names: written.sheets.join(", ") })}
            </p>
            {written.notes === "comments" && <p className="type-body text-muted">{t("notesAsComments")}</p>}
            {written.notes === "none" && <p className="type-body text-muted">{t("noNotes")}</p>}
          </div>
        )}
        {problem && <p className="type-alert" role="alert">{problem}</p>}
      </section>
      <ModelInPane
        companies={companies ?? []}
        support={{
          notes: office.context.requirements.isSetSupported("ExcelApi", "1.18"),
          comments: office.context.requirements.isSetSupported("ExcelApi", "1.10"),
        }}
      />
      <ReadInPane
        companies={companies ?? []}
        onCompanies={setCompanies}
        onAccepted={async (companyIdAccepted, ids) => {
          const found = await listCoverage();
          setCompanies(found?.companies ?? []);
          setCompanyId(companyIdAccepted);
          await importPeriods(ids);
        }}
      />
    </>
  );
}

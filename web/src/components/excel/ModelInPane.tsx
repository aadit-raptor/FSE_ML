"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { readActiveSheet } from "@/lib/excel/read";
import { writeCells, type Written } from "@/lib/excel/write";
import type { CoverageCompany } from "@/lib/research";
import {
  getSaved, putSaved, refusalKey, rollForward, saveSheet, savedCells, savedReference,
} from "@/lib/savedModel";
import { UNIT_RUPEES, blankCompany, fillFromCoverage, loadCatalogue, modelCells, type Catalogue } from "@/lib/sectorModel";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";

/**
 * "Load model" in the Excel add-in (PLAN.md R4): a sector model written into the open workbook as live formulas
 * -- the download's own cells -- with the past filled from a covered company's accepted quarters, or one of the
 * companies the model was solved by hand for. Assumptions not typed are blank yellow cells, each named on its
 * sheet: the analyst types them in Excel and the projection moves with no call to Variater.
 *
 * A covered company's model is its saved model (R5): loaded as saved (started and saved if it has none), its sheet
 * says so beside the title. "Save to Variater" reads that sheet's named cells back -- assumptions and constants --
 * and the API refuses a sheet whose structure was changed, saying what changed. "Roll forward" moves the saved
 * model on to newer accepted results and writes it again as new sheets.
 *
 * i18n-keys: models.sector_*
 * i18n-keys: models.example_*
 * i18n-keys: models.unit_*
 * i18n-keys: models.saved_*
 */
const select = "border border-line bg-field px-2 py-1.5 text-[12px] text-ink";

type Outcome = { tone: "status" | "alert"; text: string; detail?: string };

export function ModelInPane({ companies, support }: { companies: CoverageCompany[]; support: { notes: boolean; comments: boolean } }) {
  const t = useTranslations("excel");
  const words = useTranslations("models");
  const startTip = useTip("modelStartFrom");
  const sectorTip = useTip("modelSector");
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [from, setFrom] = useState("");
  const [sector, setSector] = useState("");
  const [writing, setWriting] = useState(false);
  const [written, setWritten] = useState<Written | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);

  useEffect(() => {
    let cancelled = false;
    void loadCatalogue().then((found) => {
      if (!cancelled) setCatalogue(found);
    });
    return () => {
      cancelled = true;
    };
  }, []);
  if (!catalogue) return null;

  const sectors = catalogue.models.filter((m) => m.offered).flatMap((m) => m.sectors.map((s) => ({ sector: s, model: m })));
  const chosen = sectors.find((s) => s.sector === sector) ?? sectors[0];
  const start = from || (companies[0] ? `company:${companies[0].id}` : `example:${catalogue.examples[0]?.id ?? ""}`);
  const run = () => window.Excel?.run;

  /** The covered company's saved model, started from its accepted quarters and saved when it has none yet. */
  const savedFor = async (covered: CoverageCompany) => {
    const found = await getSaved(covered.id);
    if (!found) throw new Error("saved");
    if (found.missing) {
      if (!chosen) throw new Error("sector");
      const fill = await fillFromCoverage(covered.id, "consolidated");
      const unit = fill?.unit ?? "lakhs";
      const company = blankCompany(covered.name, chosen.sector, `${fill?.currency ?? "INR"} ${words(`unit_${unit}`)}`,
        UNIT_RUPEES[unit] ?? 1, fill);
      const made = await putSaved(covered.id, { structure: chosen.model.default, company, scope: "consolidated", as_of: fill?.as_of ?? null });
      if (!made.model) throw new Error("saved");
    }
    return savedCells(covered.id);
  };

  const load = async () => {
    setWriting(true);
    setProblem(null);
    setWritten(null);
    setOutcome(null);
    try {
      const [kind, id] = start.split(":");
      const example = catalogue.examples.find((e) => e.id === id);
      let cells;
      if (kind === "example" && example) cells = await modelCells(example.structure, example.company);
      else {
        const covered = companies.find((c) => c.id === id);
        if (!covered) throw new Error("company");
        cells = await savedFor(covered);
      }
      const excel = run();
      if (!cells || !excel) throw new Error("cells");
      setWritten(await writeCells((batch) => excel(batch), cells, support, 0));
    } catch {
      setProblem(t("modelFailed"));
    } finally {
      setWriting(false);
    }
  };

  /** The active sheet, and the saved model it says it is (null when it is no saved model's). */
  const activeSheet = async () => {
    const excel = run();
    if (!excel) throw new Error("excel");
    const sheet = await readActiveSheet((batch) => excel(batch));
    const at = sheet.names.variater_model?.split("!").pop()?.replace(/\$/g, "") ?? "B1";
    const match = /^([A-Z]+)(\d+)$/.exec(at);
    const column = match ? match[1].split("").reduce((n, ch) => n * 26 + ch.charCodeAt(0) - 64, 0) : 2;
    const reference = savedReference(sheet.grid[(match ? Number(match[2]) : 1) - 1]?.[column - 1]);
    return { sheet, reference };
  };

  const saveBack = async () => {
    setWriting(true);
    setOutcome(null);
    try {
      const { sheet, reference } = await activeSheet();
      if (!reference) {
        setOutcome({ tone: "alert", text: words("saved_other_model") });
        return;
      }
      const answer = await saveSheet(reference, sheet.grid, sheet.names);
      if (answer.saved) setOutcome({ tone: "status", text: t("savedBack", { count: answer.saved.changed.length }) });
      else setOutcome({ tone: "alert", text: words(refusalKey(answer.refused?.code ?? "failed")), detail: answer.refused?.message });
    } catch {
      setOutcome({ tone: "alert", text: words("saved_failed") });
    } finally {
      setWriting(false);
    }
  };

  const rollOn = async () => {
    setWriting(true);
    setOutcome(null);
    setWritten(null);
    try {
      const { reference } = await activeSheet();
      if (!reference) {
        setOutcome({ tone: "alert", text: words("saved_other_model") });
        return;
      }
      const answer = await rollForward(reference);
      if (!answer.rolled) {
        setOutcome({ tone: "alert", text: words(refusalKey(answer.refused?.code ?? "failed")) });
        return;
      }
      const cells = await savedCells(reference);
      const excel = run();
      if (!cells || !excel) throw new Error("cells");
      setWritten(await writeCells((batch) => excel(batch), cells, support, 0));
      const c = answer.rolled.comparison;
      setOutcome({ tone: "status", text: t("rolled", { period: c.period, lines: c.lines.length }) });
    } catch {
      setOutcome({ tone: "alert", text: words("saved_failed") });
    } finally {
      setWriting(false);
    }
  };

  return (
    <section className="grid gap-2.5" data-load-model>
      <h2 className="type-input-group">{t("modelTitle")}</h2>
      <p className="type-body text-muted">{t("modelHint")}</p>
      <label className="grid gap-1">
        <span className="type-input-label"><TipLabel tip={startTip}>{t("modelFrom")}</TipLabel></span>
        <select aria-describedby={startTip.id} value={start} onChange={(e) => setFrom(e.target.value)} className={select}>
          {companies.map((c) => (
            <option key={c.id} value={`company:${c.id}`}>{c.name}</option>
          ))}
          {catalogue.examples.map((e) => (
            <option key={e.id} value={`example:${e.id}`}>{words(`example_${e.id.replace(/-/g, "_")}`)}</option>
          ))}
        </select>
      </label>
      <TipText tip={startTip} />
      {start.startsWith("company:") && (
        <>
        <label className="grid gap-1">
          <span className="type-input-label"><TipLabel tip={sectorTip}>{t("modelSector")}</TipLabel></span>
          <select aria-describedby={sectorTip.id} value={chosen?.sector ?? ""} onChange={(e) => setSector(e.target.value)} className={select}>
            {sectors.map((s) => (
              <option key={s.sector} value={s.sector}>{words(`sector_${s.sector}`)}</option>
            ))}
          </select>
        </label>
        <TipText tip={sectorTip} />
        </>
      )}
      <div>
        <PrimaryButton disabled={writing} onClick={() => void load()}>{t("loadModel")}</PrimaryButton>
      </div>
      {writing && <p className="type-step" role="status">{t("importing")}</p>}
      {written && (
        <p className="type-body" role="status" data-model-written={written.sheets.length}>
          {t("modelWritten", { names: written.sheets.join(", ") })}
        </p>
      )}
      {problem && <p className="type-alert" role="alert">{problem}</p>}

      <h2 className="type-input-group mt-2">{t("savedTitle")}</h2>
      <p className="type-body text-muted">{t("savedHint")}</p>
      <div className="flex flex-wrap gap-2">
        <PrimaryButton disabled={writing} onClick={() => void saveBack()}>{t("saveBack")}</PrimaryButton>
        <SecondaryButton disabled={writing} onClick={() => void rollOn()}>{t("rollForward")}</SecondaryButton>
      </div>
      {outcome && (
        <p className={outcome.tone === "alert" ? "type-alert" : "type-body"} role={outcome.tone} data-saved-outcome={outcome.tone}>
          {outcome.text} {outcome.detail}
        </p>
      )}
    </section>
  );
}

"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { PrimaryButton } from "@/components/ui/Screen";
import { writeCells, type Written } from "@/lib/excel/write";
import type { CoverageCompany } from "@/lib/research";
import { UNIT_RUPEES, blankCompany, fillFromCoverage, loadCatalogue, modelCells, type Catalogue } from "@/lib/sectorModel";

/**
 * "Load model" in the Excel add-in (PLAN.md R4): a sector model written into the open workbook as live formulas
 * -- the download's own cells -- with the past filled from a covered company's accepted quarters, or one of the
 * companies the model was solved by hand for. Assumptions not typed are blank yellow cells, each named on its
 * sheet: the analyst types them in Excel and the projection moves with no call to Variater.
 *
 * i18n-keys: models.sector_*
 * i18n-keys: models.example_*
 * i18n-keys: models.unit_*
 */
const select = "border border-line bg-field px-2 py-1.5 text-[12px] text-ink";

export function ModelInPane({ companies, support }: { companies: CoverageCompany[]; support: { notes: boolean; comments: boolean } }) {
  const t = useTranslations("excel");
  const words = useTranslations("models");
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [from, setFrom] = useState("");
  const [sector, setSector] = useState("");
  const [writing, setWriting] = useState(false);
  const [written, setWritten] = useState<Written | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

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

  const load = async () => {
    setWriting(true);
    setProblem(null);
    setWritten(null);
    try {
      const [kind, id] = start.split(":");
      const example = catalogue.examples.find((e) => e.id === id);
      let body;
      if (kind === "example" && example) body = { structure: example.structure, company: example.company };
      else {
        const covered = companies.find((c) => c.id === id);
        if (!covered || !chosen) throw new Error("company");
        const fill = await fillFromCoverage(covered.id, "consolidated");
        const unit = fill?.unit ?? "lakhs";
        const company = blankCompany(covered.name, chosen.sector, `${fill?.currency ?? "INR"} ${words(`unit_${unit}`)}`,
          UNIT_RUPEES[unit] ?? 1, fill);
        body = { structure: chosen.model.default, company };
      }
      const cells = await modelCells(body.structure, body.company);
      const run = window.Excel?.run;
      if (!cells || !run) throw new Error("cells");
      setWritten(await writeCells((batch) => run(batch), cells, support, 0));
    } catch {
      setProblem(t("modelFailed"));
    } finally {
      setWriting(false);
    }
  };

  return (
    <section className="grid gap-2.5" data-load-model>
      <h2 className="type-input-group">{t("modelTitle")}</h2>
      <p className="type-body text-muted">{t("modelHint")}</p>
      <label className="grid gap-1">
        <span className="type-input-label">{t("modelFrom")}</span>
        <select value={start} onChange={(e) => setFrom(e.target.value)} className={select}>
          {companies.map((c) => (
            <option key={c.id} value={`company:${c.id}`}>{c.name}</option>
          ))}
          {catalogue.examples.map((e) => (
            <option key={e.id} value={`example:${e.id}`}>{words(`example_${e.id.replace(/-/g, "_")}`)}</option>
          ))}
        </select>
      </label>
      {start.startsWith("company:") && (
        <label className="grid gap-1">
          <span className="type-input-label">{t("modelSector")}</span>
          <select value={chosen?.sector ?? ""} onChange={(e) => setSector(e.target.value)} className={select}>
            {sectors.map((s) => (
              <option key={s.sector} value={s.sector}>{words(`sector_${s.sector}`)}</option>
            ))}
          </select>
        </label>
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
    </section>
  );
}

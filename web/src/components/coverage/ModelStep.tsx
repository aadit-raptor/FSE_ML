"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { useProfile } from "@/components/auth/ProfileProvider";
import { Notice, PrimaryButton, RailGroup, Screen, SecondaryButton } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { fmtCount } from "@/lib/format";
import { listCoverage, type CoverageCompany } from "@/lib/research";
import {
  UNIT_RUPEES, blankCompany, changeModel, downloadModel, editCompany, fillFromCoverage, loadCatalogue, setModel,
  useModelState, withValue, type Block, type Catalogue, type Change, type ModelCompany, type ModelState,
} from "@/lib/sectorModel";

import { ModelGrid } from "./ModelGrid";

/**
 * Coverage -> Model (PLAN.md R4): pick a sector (which picks the model and its words), or open one of the
 * companies the model was solved by hand for; make the last changes for the company (add, drop or swap a line of
 * revenue); type the drivers' history and the assumptions; see the projection at once; download it as live
 * Excel formulas. The model says "not yet vetted" until a practising analyst has read it.
 *
 * i18n-keys: models.block_*
 * i18n-keys: models.sector_*
 * i18n-keys: models.option_*
 * i18n-keys: models.choice_*
 * i18n-keys: models.fill_*
 * i18n-keys: models.error_*
 * i18n-keys: models.unit_*
 * i18n-keys: models.example_*
 */
const UNITS = ["lakhs", "crores", "millions", "thousands", "billions", "units"];
const ERRORS = ["model_refused", "company_refused", "cannot_work_out"];
const field = "border border-line bg-canvas px-2 py-1 text-[11px] text-ink";

export function ModelStep() {
  const t = useTranslations("models");
  const { subject } = useProfile();
  const model = useModelState(subject);
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [companies, setCompanies] = useState<CoverageCompany[]>([]);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([loadCatalogue(), listCoverage()]).then(([found, coverage]) => {
      if (cancelled) return;
      setCatalogue(found);
      setCompanies(coverage?.companies ?? []);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const run = model.run;
  const edit = (company: ModelCompany) => editCompany(subject, company);

  return (
    <Screen
      rail={
        <>
          <StartGroup catalogue={catalogue} companies={companies} owner={subject} />
          {model.structure && model.company && run && catalogue && (
            <>
              <LinesGroup model={model} catalogue={catalogue} owner={subject} />
              <CompanyGroup model={model} onChange={edit} />
              <RailGroup title={t("workbookGroup")}>
                <p className="type-body mb-1.5 text-[10.5px]">{t("workbookHint")}</p>
                <PrimaryButton onClick={() => void downloadModel(model.structure!, model.company!)}>{t("download")}</PrimaryButton>
              </RailGroup>
            </>
          )}
        </>
      }
    >
      {run && !run.vetted && (
        <Notice tone="attention" title={t("notVettedTitle")} role="note">
          {t("notVetted")}
        </Notice>
      )}
      {model.refused && (
        <Notice tone="loss" title={t("refusedTitle")} role="alert">
          {t(`error_${ERRORS.includes(model.refused.code) ? model.refused.code : "failed"}`)} {model.refused.message}
        </Notice>
      )}
      {run && run.missing.length > 0 && (
        <Notice tone="info" title={t("missingTitle")}>{t("missing", { count: run.missing.length })}</Notice>
      )}
      {run && run.undefined.length > 0 && (
        <Notice tone="info" title={t("undefinedTitle")}>{t("undefined", { count: run.undefined.length })}</Notice>
      )}
      {run && run.dropped.length > 0 && (
        <Notice tone="attention" title={t("droppedTitle")}>{t("dropped", { count: run.dropped.length })}</Notice>
      )}
      {model.fillNotes.map((note) => (
        <Notice key={note} tone="info" title={t("fillTitle")}>{t(`fill_${note}`)}</Notice>
      ))}
      <Tiles>
        {!run && (
          <Tile span={12} title={t("emptyTitle")}>
            <p className="type-body text-[10.5px]">{t("emptyBody")}</p>
          </Tile>
        )}
        {run && model.company && (
          <Tile span={12} title={t("gridTitle", { name: model.company.name, model: run.name })} unit={model.company.unit_name}>
            <ModelGrid run={run} onEdit={(row, period, value) => edit(withValue(model.company!, row, period, value))} />
            <ul className="mt-2 grid gap-0.5">
              {run.notes.map((note) => (
                <li key={note} className="type-body text-[10px] text-muted">{note}</li>
              ))}
            </ul>
          </Tile>
        )}
      </Tiles>
    </Screen>
  );
}

function StartGroup({ catalogue, companies, owner }: { catalogue: Catalogue | null; companies: CoverageCompany[]; owner: string | null }) {
  const t = useTranslations("models");
  const [from, setFrom] = useState("new");
  const [name, setName] = useState("");
  const [sector, setSector] = useState("");
  const [covered, setCovered] = useState("");
  const [scope, setScope] = useState<"consolidated" | "standalone">("consolidated");
  const [currency, setCurrency] = useState("INR");
  const [unit, setUnit] = useState("lakhs");
  const [starting, setStarting] = useState(false);
  if (!catalogue) return <RailGroup title={t("startGroup")}><p className="type-step">{t("loading")}</p></RailGroup>;

  const offered = catalogue.models.filter((m) => m.offered);
  const sectors = offered.flatMap((m) => m.sectors.map((s) => ({ sector: s, model: m })));
  const chosen = sectors.find((s) => s.sector === sector) ?? sectors[0];
  const companyName = name.trim() || companies.find((c) => c.id === covered)?.name || "";

  const start = async () => {
    setStarting(true);
    try {
      const example = catalogue.examples.find((e) => e.id === from);
      if (example) {
        await setModel(owner, example.structure, example.company, []);
        return;
      }
      if (!chosen) return;
      const fill = covered ? await fillFromCoverage(covered, scope) : null;
      const unitWord = t(`unit_${fill?.unit ?? unit}`);
      const company = blankCompany(companyName, chosen.sector, `${fill?.currency ?? currency} ${unitWord}`,
        UNIT_RUPEES[unit] ?? 1, fill);
      await setModel(owner, chosen.model.default, company, fill?.notes ?? []);
    } finally {
      setStarting(false);
    }
  };

  return (
    <RailGroup title={t("startGroup")}>
      <label className="grid gap-1">
        <span className="type-input-label">{t("startFrom")}</span>
        <select className={field} value={from} onChange={(e) => setFrom(e.target.value)}>
          <option value="new">{t("newCompany")}</option>
          {catalogue.examples.map((e) => (
            <option key={e.id} value={e.id}>{t(`example_${e.id.replace(/-/g, "_")}`)}</option>
          ))}
        </select>
      </label>
      {from === "new" && (
        <>
          <label className="grid gap-1">
            <span className="type-input-label">{t("sector")}</span>
            <select className={field} value={chosen?.sector ?? ""} onChange={(e) => setSector(e.target.value)}>
              {sectors.map((s) => (
                <option key={s.sector} value={s.sector}>{t(`sector_${s.sector}`)}</option>
              ))}
            </select>
          </label>
          <label className="grid gap-1">
            <span className="type-input-label">{t("coveredCompany")}</span>
            <select className={field} value={covered} onChange={(e) => setCovered(e.target.value)}>
              <option value="">{t("noCoveredCompany")}</option>
              {companies.map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </label>
          {covered && (
            <label className="grid gap-1">
              <span className="type-input-label">{t("scope")}</span>
              <select className={field} value={scope} onChange={(e) => setScope(e.target.value as typeof scope)}>
                <option value="consolidated">{t("scopeConsolidated")}</option>
                <option value="standalone">{t("scopeStandalone")}</option>
              </select>
            </label>
          )}
          <label className="grid gap-1">
            <span className="type-input-label">{t("companyName")}</span>
            <input className={field} value={name} maxLength={200} onChange={(e) => setName(e.target.value)} />
          </label>
          {!covered && (
            <div className="grid grid-cols-2 gap-1.5">
              <label className="grid gap-1">
                <span className="type-input-label">{t("currency")}</span>
                <input className={field} value={currency} maxLength={12} onChange={(e) => setCurrency(e.target.value)} />
              </label>
              <label className="grid gap-1">
                <span className="type-input-label">{t("moneyUnit")}</span>
                <select className={field} value={unit} onChange={(e) => setUnit(e.target.value)}>
                  {UNITS.map((u) => (
                    <option key={u} value={u}>{t(`unit_${u}`)}</option>
                  ))}
                </select>
              </label>
            </div>
          )}
        </>
      )}
      <div className="mt-1.5">
        <PrimaryButton disabled={starting || (from === "new" && !companyName)} onClick={() => void start()}>
          {t("start")}
        </PrimaryButton>
      </div>
    </RailGroup>
  );
}

function LinesGroup({ model, catalogue, owner }: { model: ModelState; catalogue: Catalogue; owner: string | null }) {
  const t = useTranslations("models");
  const [adding, setAdding] = useState<Block>("growth");
  const structure = model.structure!;
  const counting = new Set(catalogue.blocks.filter((b) => b.counts_places).map((b) => b.id));
  const change = async (action: Change, key: string | null, block: Block | null) => {
    const { structure: next } = await changeModel(structure, action, key, block);
    if (next && model.company) await setModel(owner, next, model.company);
  };
  return (
    <RailGroup title={t("linesGroup")}>
      {structure.segments.map((seg, i) => (
        <div key={seg.key} className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-1.5" data-line={seg.key}>
          <label className="grid min-w-0 gap-1">
            <span className="type-input-label">{t("line", { n: String(i + 1) })}</span>
            <select className={field} value={seg.block} onChange={(e) => void change("swap", seg.key, e.target.value as Block)}>
              {catalogue.blocks.map((b) => (
                <option key={b.id} value={b.id}>{t(`block_${b.id}`)}</option>
              ))}
            </select>
          </label>
          <SecondaryButton disabled={structure.segments.length === 1} onClick={() => void change("drop", seg.key, null)}>
            {t("drop")}
          </SecondaryButton>
        </div>
      ))}
      <div className="mt-1.5 grid grid-cols-[minmax(0,1fr)_auto] items-end gap-1.5">
        <label className="grid min-w-0 gap-1">
          <span className="type-input-label">{t("addLine")}</span>
          <select className={field} value={adding} onChange={(e) => setAdding(e.target.value as Block)}>
            {catalogue.blocks.map((b) => (
              <option key={b.id} value={b.id}>{t(`block_${b.id}`)}</option>
            ))}
          </select>
        </label>
        <SecondaryButton disabled={structure.segments.length >= 8} onClick={() => void change("add", null, adding)}>
          {t("add")}
        </SecondaryButton>
      </div>
      <label className="mt-1.5 grid gap-1">
        <span className="type-input-label">{t("places")}</span>
        <select className={field} value={structure.places ?? ""} onChange={(e) => void change("places", e.target.value || null, null)}>
          <option value="">{t("placesNone")}</option>
          {structure.segments.map((seg, i) =>
            counting.has(seg.block) ? (
              <option key={seg.key} value={seg.key}>{t("line", { n: String(i + 1) })}</option>
            ) : null,
          )}
        </select>
      </label>
    </RailGroup>
  );
}

function CompanyGroup({ model, onChange }: { model: ModelState; onChange: (company: ModelCompany) => void }) {
  const t = useTranslations("models");
  const company = model.company!;
  const run = model.run!;
  const line = (choice: string) => /^line(\d)_/.exec(choice)?.[1] ?? "";
  return (
    <RailGroup title={t("companyGroup")}>
      <label className="grid gap-1">
        <span className="type-input-label">{t("companyName")}</span>
        <input className={field} value={company.name} maxLength={200} onChange={(e) => onChange({ ...company, name: e.target.value })} />
      </label>
      <label className="grid gap-1">
        <span className="type-input-label">{t("unitName")}</span>
        <input className={field} value={company.unit_name} maxLength={200} onChange={(e) => onChange({ ...company, unit_name: e.target.value })} />
      </label>
      {run.constants.map((c) => (
        <label key={c.key} className="grid gap-1">
          <span className="type-input-label">{c.label}</span>
          <select className={field} value={String(company.constants?.[c.key] ?? "")}
            onChange={(e) => onChange({ ...company, constants: { ...company.constants, [c.key]: Number(e.target.value) } })}>
            {Object.entries(UNIT_RUPEES).map(([u, n]) => (
              <option key={u} value={String(n)}>{t("unitRupees", { unit: t(`unit_${u}`), count: fmtCount(n) })}</option>
            ))}
          </select>
        </label>
      ))}
      {Object.entries(run.choices).map(([choice, options]) => (
        <label key={choice} className="grid gap-1">
          <span className="type-input-label">{t(`choice_${choice.replace(/^line\d_/, "")}`, { line: line(choice) })}</span>
          <select className={field} value={run.choices_made[choice] ?? options[0]}
            onChange={(e) => onChange({ ...company, choices: { ...run.choices_made, [choice]: e.target.value } })}>
            {options.map((o) => (
              <option key={o} value={o}>{t(`option_${o}`)}</option>
            ))}
          </select>
        </label>
      ))}
      {run.segments.map((seg) => {
        const n = model.structure!.segments.findIndex((s) => s.key === seg) + 1;
        const off = (company.off ?? []).includes(seg);
        return (
          <label key={seg} className="mt-1 flex items-center gap-2">
            <input type="checkbox" checked={off}
              onChange={() => onChange({ ...company, off: off ? (company.off ?? []).filter((s) => s !== seg) : [...(company.off ?? []), seg] })} />
            <span className="type-input-label">{t("lineOff", { n: String(n) })}</span>
          </label>
        );
      })}
    </RailGroup>
  );
}

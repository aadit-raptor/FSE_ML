"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { Notice, PrimaryButton, RailGroup, SecondaryButton } from "@/components/ui/Screen";
import { Tile } from "@/components/ui/Tile";
import { fmtNumber, fmtRate } from "@/lib/format";
import { formatDateTime } from "@/lib/locale";
import { addCompany } from "@/lib/research";
import {
  downloadSaved, keepVersion, listVersions, refusalKey, restoreVersion, rollForward, savedFrom,
  type ModelVersion, type RollAnswer, type RollComparison,
} from "@/lib/savedModel";
import { flushSave, keepAs, setModel, workbookFilename, type ModelState } from "@/lib/sectorModel";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";

/**
 * A covered company's saved model (PLAN.md R5): saved as it is typed, its versions kept and restorable, rolled
 * forward when newer results are accepted -- the model and assumptions kept, the past moved on, and the estimate
 * made before the results set beside them, as it stood when the model rolled (later edits never change it).
 *
 * i18n-keys: models.saved_*
 * i18n-keys: models.versionKind_*
 * i18n-keys: models.basis_*
 * i18n-keys: models.rolled*
 */
const field = "border border-line bg-canvas px-2 py-1 text-[11px] text-ink";

export function useVersions(companyId: string | null, version: number | null): [ModelVersion[], () => void] {
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [asked, setAsked] = useState(0);
  useEffect(() => {
    if (!companyId) return;
    let cancelled = false;
    void listVersions(companyId).then((found) => {
      if (!cancelled) setVersions(found);
    });
    return () => {
      cancelled = true;
    };
  }, [companyId, version, asked]);
  return [companyId ? versions : [], () => setAsked((n) => n + 1)];
}

export function SavedGroup({ model, owner, versions, reload }: {
  model: ModelState; owner: string | null; versions: ModelVersion[]; reload: () => void;
}) {
  const t = useTranslations("models");
  const [label, setLabel] = useState("");
  const labelTip = useTip("modelVersionLabel");
  const [busy, setBusy] = useState(false);
  const saved = model.saved;
  const company = model.company!;

  if (!saved) {
    const keep = async () => {
      setBusy(true);
      try {
        const { company: made } = await addCompany(company.name);
        if (made) keepAs({ companyId: made.id, scope: "consolidated", asOf: null, version: 0, updatedAt: null, newer: null });
      } finally {
        setBusy(false);
      }
    };
    return (
      <RailGroup title={t("savedGroup")}>
        <p className="type-body mb-1.5 text-[10.5px]">{t("notSaved")}</p>
        <PrimaryButton disabled={busy || !company.name.trim()} onClick={() => void keep()}>{t("keepInCoverage")}</PrimaryButton>
      </RailGroup>
    );
  }

  const restore = async (number: number) => {
    setBusy(true);
    try {
      const back = await restoreVersion(saved.companyId, number);
      if (back) await setModel(owner, back.structure, back.company, [], savedFrom(back));
      reload();
    } finally {
      setBusy(false);
    }
  };
  const keep = async () => {
    setBusy(true);
    try {
      if (await keepVersion(saved.companyId, label.trim() || null)) setLabel("");
      reload();
    } finally {
      setBusy(false);
    }
  };

  return (
    <RailGroup title={t("savedGroup")}>
      <p className="type-step text-[10.5px]" role="status" data-save-state={model.saving}>
        {model.saving === "saving" ? t("saving") : model.saving === "failed" ? t(refusalKey(model.saveRefused?.code ?? "failed"))
          : t("savedAt", { version: String(saved.version) })}
      </p>
      <label className="grid gap-1">
        <span className="type-input-label"><TipLabel tip={labelTip}>{t("versionLabel")}</TipLabel></span>
        <input aria-describedby={labelTip.id} className={field} value={label} maxLength={120} onChange={(e) => setLabel(e.target.value)} />
      </label>
      <TipText tip={labelTip} />
      <div className="mt-1 flex flex-wrap gap-1.5">
        <SecondaryButton disabled={busy} onClick={() => void keep()}>{t("keepVersion")}</SecondaryButton>
        <SecondaryButton onClick={() => void downloadSaved(saved.companyId, workbookFilename(company, model.structure!.model))}>
          {t("download")}
        </SecondaryButton>
      </div>
      <ul className="mt-2 grid gap-1" aria-label={t("versionsTitle")}>
        {versions.slice(0, 12).map((v) => (
          <li key={v.number} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-1.5" data-version={v.number}>
            <span className="type-body min-w-0 truncate text-[10.5px]">
              {t("versionLine", { number: String(v.number), kind: t(`versionKind_${v.kind}`), label: v.label ?? "", at: formatDateTime(new Date(v.created_at)) })}
            </span>
            <SecondaryButton disabled={busy} onClick={() => void restore(v.number)}>{t("restore")}</SecondaryButton>
          </li>
        ))}
      </ul>
    </RailGroup>
  );
}

/** "Newer results are accepted": the roll forward, and what it carried or left to type. */
export function RollNotice({ model, owner, onRolled }: { model: ModelState; owner: string | null; onRolled: () => void }) {
  const t = useTranslations("models");
  const [busy, setBusy] = useState(false);
  const [rolled, setRolled] = useState<RollAnswer | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const saved = model.saved;
  const roll = async () => {
    if (!saved) return;
    setBusy(true);
    setProblem(null);
    try {
      await flushSave();                  // the roll reads the saved copy: an edit just typed must be in it
      const answer = await rollForward(saved.companyId);
      if (answer.rolled) {
        setRolled(answer.rolled);
        const m = answer.rolled.model;
        await setModel(owner, m.structure, m.company, [], savedFrom(m));
        onRolled();
      } else setProblem(refusalKey(answer.refused?.code ?? "failed"));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      {saved?.newer && (
        <Notice tone="attention" title={t("newerTitle")} role="note"
          actions={<PrimaryButton disabled={busy} onClick={() => void roll()}>{t("rollForward")}</PrimaryButton>}>
          {t("newer", { end: saved.newer.ytd_end ?? saved.newer.ly_end, months: String(saved.newer.ytd_months || 12) })}
        </Notice>
      )}
      {problem && <Notice tone="loss" title={t("rollRefusedTitle")} role="alert">{t(problem)}</Notice>}
      {rolled && (
        <Notice tone="info" title={t("rolledTitle")} role="status">
          {t(rolled.kind === "year" ? "rolledYear" : "rolledQuarter", { carried: rolled.carried.length, toType: rolled.to_type.length })}
        </Notice>
      )}
    </>
  );
}

/** The estimate made before results, beside them, line by line: the newest roll's by default. */
export function ComparisonTile({ versions }: { versions: ModelVersion[] }) {
  const pickTip = useTip("modelComparison");
  const t = useTranslations("models");
  const rolls = versions.filter((v) => v.kind === "rolled" && v.comparison);
  const [picked, setPicked] = useState<number | null>(null);
  const chosen = rolls.find((v) => v.number === picked) ?? rolls[0];
  if (!chosen?.comparison) return null;
  const c: RollComparison = chosen.comparison;
  return (
    <Tile span={12} title={t("comparisonTitle", { period: c.period })}>
      <div className="mb-1.5 flex flex-wrap items-end gap-3">
        {rolls.length > 1 && (
          <>
          <label className="grid gap-1">
            <span className="type-input-label"><TipLabel tip={pickTip}>{t("comparisonPick")}</TipLabel></span>
            <select aria-describedby={pickTip.id} className={field} value={chosen.number} onChange={(e) => setPicked(Number(e.target.value))}>
              {rolls.map((v) => (
                <option key={v.number} value={v.number}>{v.comparison!.period}</option>
              ))}
            </select>
          </label>
          <TipText tip={pickTip} />
          </>
        )}
        <p className="type-body text-[10.5px] text-muted">{t(`basis_${c.basis}`, { from: c.estimate_label })}</p>
      </div>
      <table className="w-full border-collapse font-mono text-[10.5px]" data-comparison={c.period_end}>
        <thead>
          <tr className="border-b border-line">
            <th className="type-input-label px-2 py-1 text-start">{t("row")}</th>
            <th className="type-input-label px-2 py-1 text-end">{t("estimate")}</th>
            <th className="type-input-label px-2 py-1 text-end">{t("actual")}</th>
            <th className="type-input-label px-2 py-1 text-end">{t("difference")}</th>
            <th className="type-input-label px-2 py-1 text-end">{t("differenceShare")}</th>
          </tr>
        </thead>
        <tbody>
          {c.lines.map((line) => (
            <tr key={line.row} className="border-b border-line/40" data-compare={line.row}>
              <td className="px-2 py-0.5 text-ink">{line.label}</td>
              <td className="px-2 py-0.5 text-end" data-estimate>{fmtNumber(line.estimate, 2)}</td>
              <td className="px-2 py-0.5 text-end text-bright" data-actual>{fmtNumber(line.actual, 2)}</td>
              <td className={`px-2 py-0.5 text-end ${line.difference < 0 ? "text-loss" : "text-gain"}`}>{fmtNumber(line.difference, 2, true)}</td>
              <td className="px-2 py-0.5 text-end">{fmtRate(line.share, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Tile>
  );
}

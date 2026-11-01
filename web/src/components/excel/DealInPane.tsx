"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";
import { dealCells, dealOfSheet, listDeals, removeSheet, type DealCells, type DealListed } from "@/lib/excel/deal";
import { readActiveSheet } from "@/lib/excel/read";
import { writeCells } from "@/lib/excel/write";
import { fmtMultiple, fmtRate } from "@/lib/format";

/**
 * A saved deal in the Excel add-in (PLAN.md 7.3b): "Open in Excel" runs the saved deal and writes it into the open
 * workbook as the live-formula model -- the Summary step's "Live workbook" download, cell for cell -- and says the
 * app's IRR and MOIC beside it. "Refresh" reads which deal the active sheet is, runs that deal as it is saved now
 * (an edit made on the site since) and writes the sheet over under the same name.
 */
const select = "border border-line bg-field px-2 py-1.5 text-[12px] text-ink";

type Outcome = { tone: "status" | "alert"; text: string };

export function DealInPane({ support }: { support: { notes: boolean; comments: boolean } }) {
  const t = useTranslations("excel");
  const dealTip = useTip("paneDeal");
  const [deals, setDeals] = useState<DealListed[] | null>(null);
  const [dealId, setDealId] = useState("");
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);

  useEffect(() => {
    let cancelled = false;
    void listDeals().then((found) => {
      if (!cancelled) setDeals(found ?? []);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const chosen = (deals ?? []).find((d) => d.id === dealId) ?? deals?.[0] ?? null;
  const run = () => window.Excel?.run;
  const result = (cells: DealCells) => t("dealResult", { irr: fmtRate(cells.irr, 2), moic: fmtMultiple(cells.moic) });

  const open = async () => {
    if (!chosen) return;
    setBusy(true);
    setOutcome(null);
    try {
      const cells = await dealCells(chosen.id);
      const excel = run();
      if (!cells || !excel) throw new Error("cells");
      const written = await writeCells((batch) => excel(batch), cells, support, 0);
      setOutcome({ tone: "status", text: `${t("dealWritten", { names: written.sheets.join(", ") })} ${result(cells)}` });
    } catch {
      setOutcome({ tone: "alert", text: t("dealFailed") });
    } finally {
      setBusy(false);
    }
  };

  const refresh = async () => {
    setBusy(true);
    setOutcome(null);
    try {
      const excel = run();
      if (!excel) throw new Error("excel");
      const sheet = await readActiveSheet((batch) => excel(batch));
      const id = dealOfSheet(sheet);
      if (!id) {
        setOutcome({ tone: "alert", text: t("dealNotASheet") });
        return;
      }
      const cells = await dealCells(id);
      if (!cells) throw new Error("cells");
      // the model sheet alone, under the name it had: the About sheet stays as it was
      const removed = await removeSheet((batch) => excel(batch), sheet.title);
      const model = { ...cells.sheets[0], title: removed ? sheet.title : cells.sheets[0].title };
      await writeCells((batch) => excel(batch), { sheets: [model] }, support, 0);
      setOutcome({ tone: "status", text: `${t("dealRefreshed", { name: cells.name })} ${result(cells)}` });
    } catch {
      setOutcome({ tone: "alert", text: t("dealFailed") });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="grid gap-2.5" data-deal-in-pane>
      <h2 className="type-input-group">{t("dealTitle")}</h2>
      <p className="type-body text-muted">{t("dealHint")}</p>
      {deals !== null && deals.length === 0 && <p className="type-body">{t("noDeals")}</p>}
      {deals !== null && deals.length > 0 && (
        <>
          <label className="grid gap-1">
            <span className="type-input-label"><TipLabel tip={dealTip}>{t("deal")}</TipLabel></span>
            <select aria-describedby={dealTip.id} value={chosen?.id ?? ""} onChange={(e) => setDealId(e.target.value)} className={select}>
              {deals.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </label>
          <TipText tip={dealTip} />
        </>
      )}
      <div className="flex flex-wrap gap-2">
        <PrimaryButton disabled={busy || !chosen} onClick={() => void open()}>{t("openDeal")}</PrimaryButton>
        <SecondaryButton disabled={busy} onClick={() => void refresh()}>{t("refreshDeal")}</SecondaryButton>
      </div>
      {busy && <p className="type-step" role="status">{t("importing")}</p>}
      {outcome && (
        <p className={outcome.tone === "alert" ? "type-alert" : "type-body"} role={outcome.tone} data-deal-outcome={outcome.tone}>
          {outcome.text}
        </p>
      )}
    </section>
  );
}

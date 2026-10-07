"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { PrimaryButton } from "@/components/ui/Screen";
import { Tile } from "@/components/ui/Tile";
import { fmtCount, fmtPct } from "@/lib/format";
import { FEE_SETTINGS, fetchSourcedFees, type SourcedFees } from "@/lib/library";
import { stepHref } from "@/lib/nav";

import { useSettings } from "./SettingsProvider";

/**
 * Settings -> Fees (PLAN.md 4.5b): transaction fees, financing fees and senior amortisation from the
 * approved reference transactions' filings -- each the median across the deals that give it -- beside the
 * values in force. "Use sourced figures" applies them as Settings; until then nothing moves.
 *
 * i18n-keys: settings.tx_fee_pct, settings.fin_fee_pct, settings.def_senior_amort
 */
export function SourcedFeesTile() {
  const t = useTranslations("settings");
  const { effective, set } = useSettings();
  const [fees, setFees] = useState<SourcedFees | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    void fetchSourcedFees().then((f) => live && setFees(f));
    return () => {
      live = false;
    };
  }, []);

  const title = t("tileSourcedFees");
  if (fees === undefined) return <Tile span={6} title={title}><p className="type-body text-[9px]">{t("sourcedFeesLoading")}</p></Tile>;
  const offered = FEE_SETTINGS.filter((k) => fees?.settings[k]);
  if (!fees?.enabled || !offered.length) {
    return (
      <Tile span={6} title={title}>
        <p className="type-body text-[9.5px]" data-testid="sourced-fees-none">
          {!fees ? t("sourcedFeesUnreachable") : !fees.enabled ? t("sourcedFeesOff") : t("sourcedFeesNone", { min: String(fees.min_deals) })}
        </p>
      </Tile>
    );
  }
  const apply = () => {
    offered.forEach((k) => set(k, fees.settings[k]!.value));
  };
  return (
    <Tile span={6} title={title} action={<PrimaryButton onClick={apply}>{t("useSourcedFees")}</PrimaryButton>}>
      <table className="w-full border-collapse font-mono text-[11px]" aria-label={title}>
        <thead>
          <tr className="type-input-label">
            <th scope="col" className="px-1 py-1 text-start font-normal">{t("colSetting")}</th>
            <th scope="col" className="px-1 py-1 text-end font-normal">{t("colSourced")}</th>
            <th scope="col" className="px-1 py-1 text-end font-normal">{t("colInForce")}</th>
            <th scope="col" className="px-1 py-1 text-start font-normal">{t("colBasis")}</th>
          </tr>
        </thead>
        <tbody>
          {FEE_SETTINGS.map((k) => {
            const s = fees.settings[k];
            const now = effective[k];
            return (
              <tr key={k} data-sourced-fee={k} className="align-top">
                <th scope="row" className="border-b border-grid px-1 py-1 text-start text-[10px] font-normal text-soft">{t(k)}</th>
                <td className="border-b border-grid px-1 py-1 text-end text-accent">{s ? fmtPct(s.value, 2) : t("notOffered")}</td>
                <td className="border-b border-grid px-1 py-1 text-end text-ink">{typeof now === "number" ? fmtPct(now, 2) : ""}</td>
                <td className="border-b border-grid px-1 py-1 text-[9.5px] text-dim">
                  {s
                    ? t("sourcedFeeBasis", {
                        n: fmtCount(s.n),
                        low: fmtPct(s.low, 2),
                        high: fmtPct(s.high, 2),
                        first: String(s.first_year),
                        last: String(s.last_year),
                      })
                    : t("sourcedFeesTooFew", { min: String(fees.min_deals) })}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="type-body text-[9px]">
        {t("sourcedFeesNote", { deals: fmtCount(fees.library_size) })}{" "}
        <Link href={stepHref("library", "references")} className="text-accent underline">
          {t("sourcedFeesLink")}
        </Link>
      </p>
    </Tile>
  );
}

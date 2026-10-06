"use client";

import { useTranslations } from "next-intl";

import { MoneyScope, useMoney } from "@/components/ui/MoneyScope";
import { Tile } from "@/components/ui/Tile";
import { fmtMoney } from "@/lib/format";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
import { formatDateTime } from "@/lib/locale";

import { type Company, useCompanies } from "./CompanyProvider";

/** The summary figures, in the API's order (companies/items.py SUMMARY_FIELDS). i18n-keys: companies.field_* */
const FIELDS = [
  "revenue", "operating_income", "depreciation_amortization", "ebitda", "net_income",
  "income_tax_expense", "interest_expense", "capital_expenditures",
  "cash_and_equivalents", "accounts_receivable", "inventories", "accounts_payable",
  "total_debt", "total_assets", "total_equity", "lease_cost", "lease_liability",
] as const;

/** The loaded company's figures by fiscal year, each year linked to its filing (PLAN.md 4.1b). */
export function CompanyFiguresTile() {
  const { company } = useCompanies();
  if (!company) return null;
  // The company's own currency, whatever the screen around it shows
  return (
    <MoneyScope money={company.money}>
      <Figures company={company} />
    </MoneyScope>
  );
}

function Figures({ company }: { company: Company }) {
  const t = useTranslations("companies");
  const labels = useFiscalLabels();
  const { label: mu } = useMoney();
  const end = company.fiscal_year_end_month ?? 12;
  const yearLabel = (y: number) => labels.history(1, { endMonth: end, year: y })[0];
  const title = t("tileTitle", { company: company.company.name });
  return (
    <Tile span={12} title={title} unit={mu}>
      {company.years.length === 0 ? (
        <p className="type-body">{t("noYears")}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse font-mono text-[11px]" aria-label={title}>
            <thead>
              <tr>
                <th scope="col" className="sr-only">
                  {t("figure")}
                </th>
                {company.years.map((y) => (
                  <th key={y.fiscal_year} scope="col" className="px-2 py-1 text-end align-bottom font-normal">
                    <span className="type-input-label block">{yearLabel(y.fiscal_year)}</span>
                    {y.filing.url ? (
                      <a
                        href={y.filing.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block text-[10px] text-accent underline"
                        aria-label={t("filingLink", { year: yearLabel(y.fiscal_year) })}
                      >
                        {filed(t, y.filing)}
                      </a>
                    ) : (
                      <span className="block text-[10px] text-dim">{filed(t, y.filing)}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {FIELDS.map((f) => (
                <tr key={f} className="border-t border-line">
                  <th scope="row" className="type-input-label px-2 py-0.5 text-start font-normal">
                    {t(`field_${f}`)}
                  </th>
                  {company.years.map((y) => (
                    <td key={y.fiscal_year} className="px-2 py-0.5 text-end text-ink">
                      {fmtMoney(y.figures[f])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="font-mono text-[10px] text-dim">
        {t("attribution", { source: t(`source_${company.company.source}`), licence: company.licence })}
        {company.refreshed_at && ` ${t("refreshed", { date: formatDateTime(new Date(company.refreshed_at)) })}`}
      </p>
    </Tile>
  );
}

function filed(t: ReturnType<typeof useTranslations<"companies">>, filing: Company["years"][number]["filing"]): string {
  const form = filing.form || t("filing");
  return filing.filed_on
    ? t("filed", { form, date: formatDateTime(new Date(filing.filed_on), { dateStyle: "medium", timeZone: "UTC" }) })
    : form;
}

"use client";

import { useTranslations } from "next-intl";
import { usePathname, useRouter } from "next/navigation";

import { useDeal } from "@/components/deal/DealProvider";
import { useForecast } from "@/components/forecast/ForecastProvider";
import { SecondaryButton } from "@/components/ui/Screen";
import { fmtMoney } from "@/lib/format";
import { moneyLabel } from "@/lib/money";

import { type Company, type CompanyRef, companyKey, useCompanies } from "./CompanyProvider";

/** i18n-keys: companies.source_*, companies.standard_*, companies.reason_*, companies.matched_*, companies.note_* */
const SOURCE_ID: Record<string, string> = {
  sec: "cik", esef: "lei", companies_house: "company_number", edinet: "edinet_code",
};

/** The identifier a person would recognise first: a ticker, else the source's own id */
function shownId(ref: CompanyRef): string {
  return ref.identifiers.ticker ?? ref.identifiers[SOURCE_ID[ref.source]] ?? ref.id;
}

/**
 * Company search on every source at once (PLAN.md 4.1b): results, the sources that couldn't
 * be searched, the document-upload fallback when nobody has the company, and the loaded
 * company's "Use in deal" and "Use in forecast".
 */
export function CompanyPanel() {
  const { query, setQuery, search, searchState, load, loading, loadError, company } = useCompanies();
  const t = useTranslations("companies");
  const answer = searchState.answer;
  return (
    <div className="grid gap-1.5">
      <form
        role="search"
        className="grid grid-cols-[minmax(0,1fr)_auto] gap-2"
        onSubmit={(ev) => {
          ev.preventDefault();
          void search(query);
        }}
      >
        <label className="sr-only" htmlFor="company-query">
          {t("searchLabel")}
        </label>
        <input
          id="company-query"
          value={query}
          onChange={(ev) => setQuery(ev.target.value)}
          placeholder={t("placeholder")}
          autoComplete="off"
          maxLength={100}
          className="min-w-0 border border-line bg-field px-2 py-1 font-mono text-[11.5px] text-ink outline-none placeholder:text-dim focus:border-accent"
        />
        <button
          type="submit"
          disabled={searchState.status === "loading" || query.trim().length < 2}
          className="type-action-secondary px-2.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)] disabled:opacity-50"
        >
          {searchState.status === "loading" ? t("searching") : t("search")}
        </button>
      </form>
      {searchState.status === "error" && (
        <p role="alert" className="font-mono text-[10px] text-loss">
          {searchState.error}
        </p>
      )}
      {answer && (
        <div className="grid gap-1">
          <p role="status" className="font-mono text-[10px] text-dim">
            {t("results", { count: answer.results.length })}
            {answer.matched && ` · ${t(`matched_${answer.matched}`)}`}
          </p>
          {answer.results.length > 0 && (
            <ul className="grid max-h-[220px] grid-cols-[minmax(0,1fr)] overflow-y-auto border border-line" aria-label={t("resultsLabel")}>
              {answer.results.map((ref) => (
                <Result key={companyKey(ref)} ref_={ref} loading={loading === companyKey(ref)} disabled={loading !== null} onLoad={load} />
              ))}
            </ul>
          )}
          {answer.results.length === 0 && <p className="font-mono text-[10px] text-attention">{t("fallback_document_upload")}</p>}
          {answer.unavailable.length > 0 && (
            <p className="font-mono text-[10px] text-dim">
              {t("unavailable", {
                sources: answer.unavailable.map((u) => `${t(`source_${u.source}`)} (${t(`reason_${reason(u.reason)}`)})`).join(", "),
              })}
            </p>
          )}
        </div>
      )}
      {loadError && (
        <p role="alert" className="font-mono text-[10px] text-loss">
          {loadError}
        </p>
      )}
      {company && <Loaded company={company} />}
    </div>
  );
}

/** Reasons the API may give; anything new reads as "failed" */
const REASONS = ["not_configured", "not_found", "unreachable", "refused", "rate_limited", "unreadable", "failed"];
const reason = (r: string) => (REASONS.includes(r) ? r : "failed");

function Result({ ref_: ref, loading, disabled, onLoad }: { ref_: CompanyRef; loading: boolean; disabled: boolean; onLoad: (r: CompanyRef) => void }) {
  const t = useTranslations("companies");
  const meta = [t(`source_${ref.source}`), ref.country, shownId(ref)].filter(Boolean).join(" · ");
  return (
    <li className="border-b border-line last:border-b-0">
      <button
        type="button"
        onClick={() => onLoad(ref)}
        disabled={disabled || !ref.loadable}
        className="grid w-full min-w-0 gap-0.5 px-2 py-1 text-start hover:bg-raised disabled:opacity-60"
      >
        <span className="type-input-label truncate">{ref.name}</span>
        <span className="truncate font-mono text-[10px] text-dim">
          {loading ? t("loading") : ref.loadable ? meta : t("notLoadable", { source: t(`source_${ref.source}`) })}
        </span>
      </button>
    </li>
  );
}

function Loaded({ company }: { company: Company }) {
  const t = useTranslations("companies");
  const { setMoney, setFields } = useDeal();
  const { fromCompany, companyState } = useForecast();
  const router = useRouter();
  const path = usePathname();
  const years = company.years.map((y) => y.fiscal_year);
  const inputs = company.deal_inputs;
  const label = moneyLabel(company.money);
  return (
    <section className="grid gap-1 border-t border-line pt-1.5" aria-label={t("loadedLabel")}>
      <p className="type-input-label">{company.company.name}</p>
      <p className="font-mono text-[10px] text-muted">
        {[t(`standard_${company.accounting_standard}`), label, years.length ? `${years[0]}–${years.at(-1)}` : t("noYears")].join(" · ")}
      </p>
      {company.warnings.map((w) => (
        <p key={`${w.code}-${w.field ?? ""}`} className="font-mono text-[10px] text-attention">
          <CompanyWarning code={w.code} field={w.field} />
        </p>
      ))}
      <div className="flex flex-wrap gap-2 pt-1">
        <SecondaryButton
          disabled={!inputs}
          onClick={() => {
            if (!inputs) return;
            // The deal's other money is converted to the filing's unit first, so nothing else changes size
            setMoney({ currency: inputs.currency, unit: inputs.unit });
            setFields({
              ebitda: Number(inputs.ebitda.toFixed(6)),
              accounting_standard: inputs.accounting_standard,
              lease_cost: Number(inputs.lease_cost.toFixed(6)),
              lease_liability: Number(inputs.lease_liability.toFixed(6)),
            });
            if (path !== "/deal/inputs") router.push("/deal/inputs");
          }}
        >
          {t("useInDeal")}
        </SecondaryButton>
        <SecondaryButton
          disabled={companyState.status === "loading"}
          onClick={() => {
            void fromCompany(company);
            if (path !== "/forecast/historicals") router.push("/forecast/historicals");
          }}
        >
          {companyState.status === "loading" ? t("loading") : t("useInForecast")}
        </SecondaryButton>
      </div>
      {inputs ? (
        <>
          <p className="font-mono text-[10px] text-muted">
            {t("useInDealNote", { ebitda: `${fmtMoney(inputs.ebitda)} ${label}`, year: String(inputs.fiscal_year) })}
          </p>
          {inputs.notes.map((n) => (
            <p key={n} className="font-mono text-[10px] text-attention">
              {t(`note_${n}`)}
            </p>
          ))}
        </>
      ) : (
        <p className="font-mono text-[10px] text-attention">{t("noDealInputs")}</p>
      )}
      {companyState.status === "error" && (
        <p role="alert" className="font-mono text-[10px] text-loss">
          {companyState.error}
        </p>
      )}
    </section>
  );
}

/** i18n-keys: companies.warning_*, companies.field_* */
const WARNINGS = ["missing_figure", "scanned_accounts", "not_indexed_yet", "summary_only", "no_annual_figures"];

export function CompanyWarning({ code, field }: { code: string; field?: string | null }) {
  const t = useTranslations("companies");
  if (code === "missing_figure") return field ? t("warning_missing_figure", { field: t(`field_${field}`) }) : t("warning_other");
  return WARNINGS.includes(code) ? t(`warning_${code}`) : t("warning_other");
}

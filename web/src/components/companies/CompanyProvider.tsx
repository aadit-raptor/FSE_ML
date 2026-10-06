"use client";

import { useTranslations } from "next-intl";
import { createContext, useCallback, useContext, useMemo, useState } from "react";

import { api, type Schemas } from "@/lib/api/client";

export type CompanySearch = Schemas["CompanySearchResponse"];
export type CompanyRef = Schemas["CompanyRefOut"];
export type Company = Schemas["CompanyResponse"];

/** Fiscal years a company is loaded with: the forecast's history (PLAN.md 4.1b) */
export const COMPANY_YEARS = 3;

type Status = "idle" | "loading" | "error";

type CompanyContext = {
  query: string;
  setQuery: (q: string) => void;
  /** Every source at once (GET /api/companies/search) */
  search: (q: string) => Promise<void>;
  searchState: { status: Status; answer?: CompanySearch; error?: string };
  /** Read a company's filings (POST /api/companies/load) */
  load: (ref: CompanyRef) => Promise<void>;
  /** The source and id being loaded, while it loads */
  loading: string | null;
  loadError?: string;
  /** The loaded company, shared by the deal and the forecast screens */
  company: Company | null;
  clear: () => void;
};

const Ctx = createContext<CompanyContext | null>(null);

export const companyKey = (ref: { source: string; id: string }) => `${ref.source}/${ref.id}`;

function detail(err: unknown): string | undefined {
  const d = (err as { detail?: unknown } | undefined)?.detail;
  return typeof d === "string" ? d : undefined;
}

/**
 * Company search and the loaded company (PLAN.md 4.1b). Mounted in the shell, so a company
 * found on the deal is still there on the forecast. Nothing is stored here: company figures
 * are public and the API keeps them shared, with nothing about who asked.
 */
export function CompanyProvider({ children }: { children: React.ReactNode }) {
  const t = useTranslations("companies");
  const e = useTranslations("errors");
  const [query, setQuery] = useState("");
  const [searchState, setSearchState] = useState<CompanyContext["searchState"]>({ status: "idle" });
  const [loading, setLoading] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | undefined>();
  const [company, setCompany] = useState<Company | null>(null);

  const search = useCallback(
    async (q: string) => {
      const text = q.trim();
      if (text.length < 2) return;
      setSearchState({ status: "loading" });
      try {
        const { data, error } = await api.GET("/api/companies/search", { params: { query: { q: text } } });
        setSearchState(data ? { status: "idle", answer: data } : { status: "error", error: detail(error) ?? t("searchFailed") });
      } catch {
        setSearchState({ status: "error", error: e("apiUnreachable") });
      }
    },
    [t, e],
  );

  const load = useCallback(
    async (ref: CompanyRef) => {
      setLoading(companyKey(ref));
      setLoadError(undefined);
      try {
        const { data, error } = await api.POST("/api/companies/load", {
          body: { source: ref.source, id: ref.id, years: COMPANY_YEARS },
        });
        if (data) setCompany(data);
        else setLoadError(detail(error) ?? t("loadFailed"));
      } catch {
        setLoadError(e("apiUnreachable"));
      } finally {
        setLoading(null);
      }
    },
    [t, e],
  );

  const clear = useCallback(() => {
    setCompany(null);
    setLoadError(undefined);
  }, []);

  const value = useMemo(
    () => ({ query, setQuery, search, searchState, load, loading, loadError, company, clear }),
    [query, search, searchState, load, loading, loadError, company, clear],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useCompanies(): CompanyContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useCompanies must be used inside CompanyProvider");
  return ctx;
}

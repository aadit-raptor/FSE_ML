/**
 * The rows of the forecasting inputs (keys from core/forecasting.py).
 *
 * Each row is named by its own key in the `forecast` namespace of the
 * translation files, and each group by `titleKey` (PLAN.md 2.3b). A title or
 * row that names an amount carries a "{money}" placeholder, filled with the
 * company's money label (PLAN.md 2.2).
 *
 * i18n-keys: forecast.group*, forecast.h_*
 */

export type Group = { titleKey: string; rows: string[] };

export const HISTORY_GROUPS: Group[] = [
  {
    titleKey: "groupIncomeStatement",
    rows: ["h_rev", "h_cogs", "h_rd", "h_sga", "h_int_inc", "h_int_exp", "h_other", "h_tax", "h_da", "h_sbc"],
  },
  {
    titleKey: "groupBalanceSheet",
    rows: [
      "h_cash", "h_ar", "h_inv", "h_ocurr", "h_ppe", "h_nca", "h_lta",
      "h_ap", "h_ocl", "h_def", "h_ltd", "h_ncl", "h_cs", "h_re", "h_oci",
    ],
  },
  {
    titleKey: "groupCashFlow",
    rows: ["h_capex", "h_divs", "h_buybacks"],
  },
];

export const ASSUMPTION_GROUPS: Group[] = [
  { titleKey: "groupGrowthMargins", rows: ["rev_g", "gm", "rd", "sga", "tax"] },
  { titleKey: "groupCashFlowPct", rows: ["da", "sbc", "capex"] },
  { titleKey: "groupWorkingCapital", rows: ["ar_d", "inv_d", "ap_d", "ocl_pct", "def_pct", "nca_pct"] },
  { titleKey: "groupFinancing", rows: ["other_inc", "divs", "buybacks", "ltd_chg", "r_cash", "r_debt", "min_cash"] },
];
